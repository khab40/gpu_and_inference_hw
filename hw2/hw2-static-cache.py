"""Experimental fastest-path HW2 generation loop.

This is intentionally separate from `hw2_task.py`. It keeps the graded
submission stable while testing more aggressive serving-style optimizations:

- preallocated StaticCache instead of a growing DynamicCache
- preallocated output token tensor
- optional torch.compile() around the single-token decode step
- optional CUDA graph replay for fixed-shape decode

Run from the repository root:
    python hw2/hw2-static-cache.py
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import StaticCache

from utils import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    RESULTS_DIR,
    PROFILE_STEPS,
    PROMPT_LEN,
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
)


def configure_cuda() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def new_static_cache(model, batch_size: int, max_cache_len: int) -> StaticCache:
    cache = StaticCache(model.config, max_cache_len=max_cache_len)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=model.config.num_key_value_heads,
        head_dim=head_dim,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    return cache


def prefill_once(model, input_ids: torch.Tensor, cache: StaticCache):
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    outputs = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        logits_to_keep=1,
    )
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    next_position = torch.tensor([input_ids.shape[1]], device=input_ids.device)
    return next_token_id, next_position


def make_decode_step(model, compile_decode: bool):
    def decode_step(token_id: torch.Tensor, cache_position: torch.Tensor, cache: StaticCache):
        outputs = model(
            input_ids=token_id[:, None],
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        return torch.argmax(outputs.logits[:, -1, :], dim=-1)

    if not compile_decode:
        return decode_step

    try:
        return torch.compile(decode_step, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        print(f"torch.compile setup failed; using eager decode: {exc}")
        return decode_step


def static_cache_loop(model, input_ids, n_steps, compile_decode: bool = True):
    """StaticCache decode path with fixed cache storage and no per-step CPU sync."""
    generated = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)
    max_cache_len = input_ids.shape[1] + n_steps
    cache = new_static_cache(model, input_ids.shape[0], max_cache_len)
    decode_step = make_decode_step(model, compile_decode=compile_decode)

    with torch.inference_mode():
        next_token_id, cache_position = prefill_once(model, input_ids, cache)
        generated[0] = next_token_id

        for step in range(1, n_steps):
            if compile_decode and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            # Clone outside the compiled function. Inductor may wrap compiled
            # CUDA decode in CUDA graphs whose output storage is reused on the
            # next replay, so keeping the raw output across iterations is unsafe.
            next_token_id = decode_step(next_token_id, cache_position, cache).clone()
            generated[step] = next_token_id
            cache_position += 1

    return generated.detach().cpu().tolist()


def static_cache_eager_loop(model, input_ids, n_steps):
    return static_cache_loop(model, input_ids, n_steps, compile_decode=False)


def static_cache_compiled_loop(model, input_ids, n_steps):
    return static_cache_loop(model, input_ids, n_steps, compile_decode=True)


def dynamic_cache_loop(model, input_ids, n_steps):
    """Same optimization class as hw2_task.py, kept local to avoid importing it."""
    generated = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated[0] = next_token_id
        past_key_values = outputs.past_key_values

        for step in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id[:, None],
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated[step] = next_token_id
            past_key_values = outputs.past_key_values

    return generated.detach().cpu().tolist()


def cuda_graph_static_loop(model, input_ids, n_steps):
    """CUDA graph decode replay path.

    CUDA graph capture is sensitive to hidden allocations inside framework code.
    If capture is not possible, callers should fall back to StaticCache +
    torch.compile().
    """
    if n_steps <= 0:
        return []

    generated = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)
    max_cache_len = input_ids.shape[1] + n_steps
    cache = new_static_cache(model, input_ids.shape[0], max_cache_len)

    with torch.inference_mode():
        next_token_id, cache_position = prefill_once(model, input_ids, cache)
        generated[0] = next_token_id

        static_token = next_token_id.clone()
        static_position = cache_position.clone()

        # Warm up eager decode before capture.
        for _ in range(3):
            warm_cache = new_static_cache(model, input_ids.shape[0], max_cache_len)
            warm_token, warm_position = prefill_once(model, input_ids, warm_cache)
            _ = make_decode_step(model, compile_decode=False)(
                warm_token, warm_position, warm_cache
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_next_token = make_decode_step(model, compile_decode=False)(
                static_token, static_position, cache
            )

        for step in range(1, n_steps):
            static_token.copy_(next_token_id)
            static_position.copy_(cache_position)
            graph.replay()
            next_token_id = graph_next_token.clone()
            generated[step] = next_token_id
            cache_position += 1

    return generated.detach().cpu().tolist()


def time_loop(loop_fn, model, input_ids, label: str, n_steps: int = MAX_NEW_TOKENS) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    tokens = loop_fn(model, input_ids, n_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {n_steps} tokens in {elapsed:.3f}s ({n_steps / elapsed:.1f} tok/s)")
    print(f"Token preview: {tokens[:8]}")
    return elapsed


def profile_loop(loop_fn, model, input_ids, trace_name: str) -> None:
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
        loop_fn(model, input_ids, PROFILE_STEPS)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace saved to {trace_path}")


def run(args) -> None:
    configure_cuda()
    print("=" * 60)
    print("HW2: Experimental Optimal Generation")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    model = build_model(args.dtype)
    input_ids = get_input_ids()
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"prompt length: {PROMPT_LEN}, generated tokens: {MAX_NEW_TOKENS}")

    if args.include_baseline:
        print("\n--- Slow baseline ---")
        slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    else:
        slow_elapsed = None

    print("\n--- HW2 optimized DynamicCache loop ---")
    dynamic_elapsed = time_loop(dynamic_cache_loop, model, input_ids, "DynamicCache")

    print("\n--- StaticCache eager decode ---")
    static_eager_elapsed = time_loop(
        static_cache_eager_loop, model, input_ids, "StaticCache eager"
    )

    print("\n--- StaticCache compiled decode ---")
    static_compiled_elapsed = None
    try:
        static_compiled_elapsed = time_loop(
            static_cache_compiled_loop, model, input_ids, "StaticCache compiled"
        )
    except Exception as exc:
        print(f"StaticCache compiled path failed; continuing with measured paths: {exc}")

    graph_elapsed = None
    if args.cuda_graph:
        print("\n--- StaticCache CUDA graph decode ---")
        try:
            graph_elapsed = time_loop(cuda_graph_static_loop, model, input_ids, "CUDA graph")
        except Exception as exc:
            print(f"CUDA graph path failed; this is expected on some HF/PyTorch stacks: {exc}")

    if args.profile:
        print("\n--- Profile StaticCache compiled decode ---")
        profile_loop(
            static_cache_compiled_loop,
            model,
            input_ids,
            "v2_static_cache_compiled_trace.json",
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = [
        ("DynamicCache", dynamic_elapsed),
        ("StaticCache eager", static_eager_elapsed),
    ]
    if static_compiled_elapsed is not None:
        rows.append(("StaticCache compiled", static_compiled_elapsed))
    if graph_elapsed is not None:
        rows.append(("CUDA graph", graph_elapsed))
    for name, elapsed in rows:
        print(f"{name:22s}: {elapsed:7.3f}s  {MAX_NEW_TOKENS / elapsed:8.1f} tok/s")
    if slow_elapsed is not None:
        for name, elapsed in rows:
            print(f"Speedup vs slow ({name:22s}): {slow_elapsed / elapsed:6.2f}x")
    for name, elapsed in rows[1:]:
        print(f"Speedup vs DynamicCache ({name:14s}): {dynamic_elapsed / elapsed:6.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
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
        help="Export a profiler trace for StaticCache compiled decode.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Try CUDA graph replay for fixed-shape decode; falls back on failure.",
    )
    args = parser.parse_args()
    args.dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    return args


if __name__ == "__main__":
    run(parse_args())
