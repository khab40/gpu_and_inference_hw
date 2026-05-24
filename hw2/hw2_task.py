import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    generated_tokens = []
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)
        past_key_values = outputs.past_key_values

        for _ in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id[:, None],
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id)
            past_key_values = outputs.past_key_values

    return torch.cat(generated_tokens).detach().cpu().tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
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


def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    model = build_model(torch.float16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "Optimized")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
# - Used the model's KV cache. The optimized loop runs the full 1024-token
#   prompt once, then feeds only the newest token with past_key_values, avoiding
#   repeated O(sequence length) prompt recomputation on every decode step.
# - Removed per-step .item() synchronization. Token tensors stay on the GPU
#   during generation and are copied to the CPU once at the end for preview.
# - Removed per-step torch.cat() growth of generated_ids. The loop no longer
#   rebuilds an ever-longer input tensor because the KV cache owns history.
# - Loaded the optimized model in float16 to use lower-bandwidth, Tensor Core
#   friendly inference math on the target CUDA GPU.
# - Exact per-fix speedups should be filled from the target L40S/H100 run; this
#   local Mac environment has no CUDA device, so the timed benchmark cannot run
#   here.
#
# Biggest impact and why:
#
# The KV-cache change is the main win. It changes each decode step from a full
# forward over the entire growing sequence into a single-token forward that
# reuses stored keys and values, so the amount of attention and MLP work per
# generated token drops dramatically.
