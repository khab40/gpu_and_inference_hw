"""Second-pass HW2 optimization experiment.

This keeps the practical DynamicCache approach from the graded solution and
adds lower-risk improvements:

- `logits_to_keep=1` to avoid returning full-sequence logits
- preallocated generated-token tensor
- warmup before timed runs
- optional profiler trace

It deliberately avoids StaticCache, CUDA graphs, and importing `hw2_task.py`.

Run from the repository root:
    python hw2/hw2-dynamic-cache-v2.py --include-baseline --profile
"""

from __future__ import annotations

import argparse
import time

import torch

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


def dynamic_cache_loop_v2(model, input_ids, n_steps):
    generated = torch.empty(n_steps, device=input_ids.device, dtype=torch.long)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated[0] = next_token_id
        past_key_values = outputs.past_key_values

        for step in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated[step] = next_token_id
            past_key_values = outputs.past_key_values

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
    print("HW2: DynamicCache Optimized V2")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    model = build_model(args.dtype)
    input_ids = get_input_ids()
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")

    if args.include_baseline:
        print("\n--- Slow baseline ---")
        slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    else:
        slow_elapsed = None

    print(f"\n--- Warmup ({args.warmup_tokens} tokens) ---")
    dynamic_cache_loop_v2(model, input_ids, args.warmup_tokens)
    torch.cuda.synchronize()

    print("\n--- DynamicCache V2 ---")
    elapsed = time_loop(dynamic_cache_loop_v2, model, input_ids, "DynamicCache V2")

    if args.profile:
        print("\n--- Profile DynamicCache V2 ---")
        profile_loop(
            dynamic_cache_loop_v2,
            model,
            input_ids,
            "v2_dynamic_cache_trace.json",
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DynamicCache V2: {elapsed:7.3f}s  {MAX_NEW_TOKENS / elapsed:8.1f} tok/s")
    if slow_elapsed is not None:
        print(f"Speedup vs slow: {slow_elapsed / elapsed:6.2f}x")


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
        help="Export a profiler trace for DynamicCache V2.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="Number of generated tokens to run before timing.",
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
