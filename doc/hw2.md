# HW2: LLM Inference Optimization

## What I Do

HW2 profiles and optimizes an autoregressive generation loop for a tiny random
Llama model. The baseline repeatedly runs the full growing sequence through the
model, extracts each token to the CPU with `.item()`, and concatenates the input
tensor every step.

The implementation in `hw2/hw2_task.py` changes that loop to:

- Run the full prompt once.
- Reuse `past_key_values` for single-token decode steps.
- Keep generated token tensors on the GPU during generation.
- Copy generated tokens to the CPU only once at the end.
- Avoid rebuilding a growing `generated_ids` tensor with `torch.cat()` each
  step.
- Load the optimized model with `float16` on CUDA.
- Export profiler traces for both baseline and optimized loops.

## Run

This requires a CUDA GPU. The grading target is calibrated against L40S.

```bash
python hw2/hw2_task.py
```

Or through the Nebius helper script:

```bash
./scripts/run_hw2.sh
```

## Outputs

The run writes:

- `hw2/results/v0_slow_trace.json`
- `hw2/results/v1_optimized_trace.json`
- `hw2/results/hw2_run.log` when run through `scripts/run_hw2.sh`

Open the trace JSON files at `https://ui.perfetto.dev` to compare baseline and
optimized execution.

## Decode Optimization

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#ffffff", "primaryColor": "#e8f5ff", "primaryTextColor": "#111827", "primaryBorderColor": "#2563eb", "lineColor": "#374151", "tertiaryColor": "#f8fafc"}} }%%
flowchart TD
    A["Prompt tokens length 1024"] --> B["One full forward pass"]
    B --> C["past_key_values cache"]
    B --> D["Next token"]
    D --> E["Single-token forward"]
    C --> E
    E --> F["Updated past_key_values"]
    E --> G["Next token tensor"]
    F --> E
    G --> E
    G --> H["Copy generated tokens to CPU once"]
```

## Baseline vs Optimized

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#ffffff", "primaryColor": "#e8f5ff", "primaryTextColor": "#111827", "primaryBorderColor": "#2563eb", "lineColor": "#374151", "tertiaryColor": "#f8fafc"}} }%%
flowchart LR
    subgraph Baseline
        B1["Full sequence forward every step"] --> B2["GPU to CPU .item sync"]
        B2 --> B3["torch.cat grows input"]
        B3 --> B1
    end

    subgraph Optimized
        O1["Prompt forward once"] --> O2["Decode one token with KV cache"]
        O2 --> O3["Keep token on GPU"]
        O3 --> O2
    end
```

## Main Ideas

- KV caching changes decode from recomputing the full prompt every step to
  processing one token at a time.
- Avoiding `.item()` inside the loop removes a repeated CPU-GPU synchronization.
- Avoiding per-step concatenation removes repeated tensor allocation and copy
  work.
- The exact speedup should be recorded from the Nebius GPU run in the writeup
  block in `hw2/hw2_task.py`.
