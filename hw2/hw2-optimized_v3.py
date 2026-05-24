"""Third-pass HW2 optimization experiment.

This keeps the DynamicCache strategy that won in the collected H100 runs, but
tightens the benchmark and trims per-token overhead:

- returns CUDA tensors from the timed loop, copying to CPU only after timing
- uses repeat/median timing instead of a single sample
- warms up with a longer decode before measurement
- requests tuple outputs with `return_dict=False`
- keeps `logits_to_keep=1` and preallocated generated-token storage
- adds a custom tiny-Llama path with preallocated KV tensors and fewer
  Transformers objects in the per-token loop

It deliberately avoids changing `hw2_task.py`.

Run from the repository root:
    python hw2/hw2-optimized_v3.py --include-baseline --profile
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

from utils import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
)


def configure_cuda() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(query_states, key_states, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states, key_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def dynamic_cache_loop_v3(model, input_ids, n_steps):
    generated = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)
    if n_steps <= 0:
        return generated

    with torch.inference_mode():
        logits, past_key_values = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
            return_dict=False,
        )[:2]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated[0] = next_token_id

        for step in range(1, n_steps):
            logits, past_key_values = model(
                input_ids=next_token_id[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=False,
            )[:2]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            generated[step] = next_token_id

    return generated


class CustomLlamaKVDecoder:
    """Minimal decode path specialized to the tiny Llama model used in HW2."""

    def __init__(self, model, input_ids, max_new_tokens: int):
        self.model = model
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens

        config = model.config
        self.batch_size = input_ids.shape[0]
        self.prompt_len = input_ids.shape[1]
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.max_seq_len = self.prompt_len + max_new_tokens

        first_weight = model.model.embed_tokens.weight
        device = first_weight.device
        dtype = first_weight.dtype

        cache_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.max_seq_len,
            self.head_dim,
        )
        self.key_cache = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in model.model.layers
        ]
        self.value_cache = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in model.model.layers
        ]
        self.generated = torch.empty(max_new_tokens, device=device, dtype=torch.long)
        self.prompt_positions = torch.arange(
            0,
            self.prompt_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        self.decode_positions = torch.arange(
            self.prompt_len,
            self.prompt_len + max_new_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)

    def _forward(self, input_ids, position_ids, start_pos: int, is_prefill: bool) -> torch.Tensor:
        hidden_states = self.model.model.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]
        end_pos = start_pos + seq_len
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.model.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            hidden_shape = (
                self.batch_size,
                seq_len,
                -1,
                self.head_dim,
            )
            query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rope(query_states, key_states, cos, sin)

            self.key_cache[layer_idx][:, :, start_pos:end_pos, :] = key_states
            self.value_cache[layer_idx][:, :, start_pos:end_pos, :] = value_states
            full_key_states = self.key_cache[layer_idx][:, :, :end_pos, :]
            full_value_states = self.value_cache[layer_idx][:, :, :end_pos, :]
            full_key_states = repeat_kv(full_key_states, self.num_key_value_groups)
            full_value_states = repeat_kv(full_value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                query_states,
                full_key_states,
                full_value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_prefill,
            )
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(
                self.batch_size,
                seq_len,
                self.hidden_size,
            )
            hidden_states = residual + layer.self_attn.o_proj(attn_output)

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.model.model.norm(hidden_states)
        return self.model.lm_head(hidden_states[:, -1:, :])

    def generate(self, n_steps: int) -> torch.Tensor:
        if n_steps < 0 or n_steps > self.max_new_tokens:
            raise ValueError(f"n_steps must be between 0 and {self.max_new_tokens}")
        if n_steps == 0:
            return self.generated[:0]

        with torch.inference_mode():
            logits = self._forward(
                self.input_ids,
                self.prompt_positions,
                start_pos=0,
                is_prefill=True,
            )
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            self.generated[0] = next_token_id

            for step in range(1, n_steps):
                logits = self._forward(
                    next_token_id[:, None],
                    self.decode_positions[:, step - 1 : step],
                    start_pos=self.prompt_len + step - 1,
                    is_prefill=False,
                )
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
                self.generated[step] = next_token_id

        return self.generated[:n_steps]


def time_loop_once(generate_fn: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = generate_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, tokens


def benchmark_loop(
    generate_fn: Callable[[], torch.Tensor],
    label: str,
    n_steps: int,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    elapsed_values = []
    last_tokens = None

    for repeat_idx in range(repeats):
        elapsed, tokens = time_loop_once(generate_fn)
        elapsed_values.append(elapsed)
        last_tokens = tokens
        print(
            f"{label} repeat {repeat_idx + 1}/{repeats}: "
            f"{elapsed:.3f}s ({n_steps / elapsed:.1f} tok/s)"
        )

    median_elapsed = statistics.median(elapsed_values)
    best_elapsed = min(elapsed_values)
    worst_elapsed = max(elapsed_values)
    assert last_tokens is not None

    preview = last_tokens.detach().cpu().tolist()[:8]
    print(
        f"{label} median: {median_elapsed:.3f}s "
        f"({n_steps / median_elapsed:.1f} tok/s)"
    )
    print(
        f"{label} best/worst: {best_elapsed:.3f}s / {worst_elapsed:.3f}s "
        f"({n_steps / best_elapsed:.1f} / {n_steps / worst_elapsed:.1f} tok/s)"
    )
    print(f"Token preview: {preview}")
    return median_elapsed, last_tokens


def profile_loop(generate_fn: Callable[[], torch.Tensor], trace_name: str) -> None:
    trace_path = RESULTS_DIR / trace_name
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        generate_fn()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace saved to {trace_path}")


def run(args) -> None:
    configure_cuda()
    print("=" * 60)
    print("HW2: DynamicCache Optimized V3")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    model = build_model(args.dtype)
    input_ids = get_input_ids()
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"warmup tokens: {args.warmup_tokens}")
    print(f"timed repeats: {args.repeats}")

    if args.include_baseline:
        print("\n--- Slow baseline ---")
        slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    else:
        slow_elapsed = None

    print(f"\n--- Warmup ({args.warmup_tokens} tokens) ---")
    dynamic_cache_loop_v3(model, input_ids, args.warmup_tokens)
    custom_decoder = CustomLlamaKVDecoder(model, input_ids, MAX_NEW_TOKENS)
    custom_decoder.generate(args.warmup_tokens)
    torch.cuda.synchronize()

    print("\n--- DynamicCache V3 ---")
    dynamic_elapsed, dynamic_tokens = benchmark_loop(
        lambda: dynamic_cache_loop_v3(model, input_ids, MAX_NEW_TOKENS),
        "DynamicCache V3",
        MAX_NEW_TOKENS,
        args.repeats,
    )

    print("\n--- CustomKV V3 ---")
    custom_elapsed, custom_tokens = benchmark_loop(
        lambda: custom_decoder.generate(MAX_NEW_TOKENS),
        "CustomKV V3",
        MAX_NEW_TOKENS,
        args.repeats,
    )

    dynamic_preview = dynamic_tokens.detach().cpu().tolist()[:16]
    custom_preview = custom_tokens.detach().cpu().tolist()[:16]
    if dynamic_preview == custom_preview:
        print("CustomKV preview matches DynamicCache preview.")
    else:
        print("CustomKV preview differs from DynamicCache preview.")
        print(f"DynamicCache preview: {dynamic_preview}")
        print(f"CustomKV preview:    {custom_preview}")

    if args.profile:
        print("\n--- Profile DynamicCache V3 ---")
        profile_loop(
            lambda: dynamic_cache_loop_v3(model, input_ids, PROFILE_STEPS),
            "v3_dynamic_cache_trace.json",
        )
        print("\n--- Profile CustomKV V3 ---")
        profile_loop(
            lambda: custom_decoder.generate(PROFILE_STEPS),
            "v3_custom_kv_trace.json",
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"DynamicCache V3 median: {dynamic_elapsed:7.3f}s  "
        f"{MAX_NEW_TOKENS / dynamic_elapsed:8.1f} tok/s"
    )
    print(
        f"CustomKV V3 median:     {custom_elapsed:7.3f}s  "
        f"{MAX_NEW_TOKENS / custom_elapsed:8.1f} tok/s"
    )
    if slow_elapsed is not None:
        print(f"Dynamic speedup vs slow:{slow_elapsed / dynamic_elapsed:6.2f}x")
        print(f"Custom speedup vs slow: {slow_elapsed / custom_elapsed:6.2f}x")
    print(f"CustomKV vs Dynamic:    {dynamic_elapsed / custom_elapsed:6.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype for the experimental run.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also time the original slow baseline.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a profiler trace for DynamicCache V3.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=32,
        help="Number of generated tokens to run before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed repeats; the summary reports the median.",
    )
    args = parser.parse_args()
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be non-negative")
    if args.warmup_tokens > MAX_NEW_TOKENS:
        parser.error(f"--warmup-tokens must be <= {MAX_NEW_TOKENS}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    args.dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    return args


if __name__ == "__main__":
    run(parse_args())
