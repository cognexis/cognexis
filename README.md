# Cognexis

**A Compute-Adaptive Recurrent-Depth Large Language Model**

Cognexis is a decoder-only language model architecture that separates *reasoning depth* from *parameter count*. Instead of scaling depth exclusively by stacking more transformer layers, Cognexis reuses a shared recurrent transformer block for a configurable number of iterations at inference time. The same checkpoint serves requests at different quality, latency, and compute budgets — making loop count a runtime control knob rather than a fixed architectural constant.

---

## Table of Contents

1. [Core Idea](#core-idea)
2. [Architecture](#architecture)
3. [Loop Modes](#loop-modes)
4. [Model Configurations](#model-configurations)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Quick Start](#quick-start)
7. [Project Structure](#project-structure)
8. [Specification](#specification)
9. [Safety and Observability](#safety-and-observability)
10. [Known Limitations](#known-limitations)
11. [License](#license)

---

## Core Idea

In a standard transformer, reasoning depth is hard-coupled to parameter count — deeper models need more unique weights. Cognexis decouples these by routing every sequence through:

1. A **Prelude** of conventional transformer blocks (unique parameters).
2. A **shared recurrent core** applied *N* times (one shared parameter set, reused across all *N* iterations).
3. A **Coda** of conventional transformer blocks (unique parameters).

```
input_text
  -> tokenizer.encode
  -> token_ids
  -> embedding(token_ids)
  -> prelude(hidden)          # unique params, runs once
  -> recurrent_core(hidden)   # shared params, runs N times
  -> coda(hidden)             # unique params, runs once
  -> lm_head(hidden)
  -> logits
```

The **effective depth** is:

```
effective_depth = num_prelude_blocks + N * num_recurrent_blocks + num_coda_blocks
```

Increasing `N` deepens the computation without adding new parameters. The compute-adaptive scheduler can then decide *at runtime* how much depth each request actually needs.

---

## Architecture

Cognexis follows a five-stage pipeline. Each stage is a separate, testable module.

### Tokenizer

Converts raw text into token IDs using a subword tokenizer (SentencePiece Unigram by default). The tokenizer is part of the model contract — it is bundled with every checkpoint and validated by checksum. Required special tokens: `BOS`, `EOS`, `PAD`, `UNK`, `EOD`. Instruction-tuned deployments add role markers: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|end|>`.

### Embedding

Maps token IDs to dense vectors of size `hidden_dim`. Uses **rotary positional encoding (RoPE)** applied inside attention, not learned absolute positions. Embedding weights are **weight-tied** with the LM head by default, reducing parameter count and keeping input/output token representations aligned.

### Prelude

A stack of conventional transformer blocks with unique parameters. It builds stable, contextual token representations before recurrence begins. Without a capable Prelude, the recurrent core would need to simultaneously handle early lexical processing and deep iterative refinement, increasing instability and weakening depth generalization.

### Recurrent Core

The defining Cognexis component: **one shared transformer block** applied repeatedly. Each iteration refines the hidden state under the same causal mask. Shared parameters mean:

- Gradients accumulate from all loop unrolls into the same weights.
- Checkpoints store exactly one recurrent block — never one copy per loop.
- Stability controls (residual scaling, gating, spectral monitoring) are required.

Optional input injection anchors recurrence to the original Prelude output `h0`, preventing drift toward generic fixed points at high loop counts.

### Coda

A final stack of conventional transformer blocks with unique parameters. Integrates the refined recurrent state, performs cross-token mixing, and prepares representations for the LM head. If `num_coda_blocks = 0`, recurrent output flows directly to final norm and logits — valid but quality-affecting.

### LM Head

Linear projection from hidden states to vocabulary logits. Weight-tied to the token embedding by default. Loss is standard next-token cross-entropy, with masking for PAD tokens, non-assistant prompt tokens in SFT, and packed-document boundaries.

---

## Loop Modes

Cognexis supports three loop execution modes, selectable at request time:

| Mode | Description |
|------|-------------|
| `fixed` | Every sequence runs the configured loop count. Baseline for training, debugging, and deterministic production. |
| `adaptive_sequence` | A scheduler decides when to halt for the whole sequence. Simpler execution model than token-wise. |
| `adaptive_token` | Each token position may halt independently while others continue. More compute-efficient for heterogeneous sequences; more complex cache/masking implementation. |

All modes enforce hard bounds: the scheduler may stop early, but **never exceeds `max_loops`** or the request compute budget.

### Scheduler Modes

The adaptive scheduler can be:

- **rule-based** — halts using thresholds on hidden-state delta, confidence, entropy, or budgets.
- **value-head** — halts using a small learned head predicting marginal improvement per loop.
- **hybrid** — combines rule-based hard stops with value-head gain-per-cost predictions.
- **oracle** — research-only; uses unavailable future information for analysis only.

**Recommended for production:** conservative `hybrid` with hard bounds and `min_loops` enforced before any halt decision.

---

## Model Configurations

| Model | Hidden Size | Attention Heads | Prelude Blocks | Recurrent Blocks | Coda Blocks | Max Loops (train) | Parameters |
|-------|------------|----------------|---------------|-----------------|-------------|-------------------|-----------|
| Cognexis-8B | 4096 | 32 | 8 | 1 | 8 | 12 | ~8 B |
| Cognexis-64B | 8192 | 64 | 10 | 1 | 10 | 16 | ~64 B |
| Cognexis-256B | 12288 | 96 | 12 | 1 | 12 | 20 | ~256 B |
| Cognexis-1.28T | 16384 | 128 | 16 | 1 | 16 | 24 | ~1.28 T |

The recurrent core always uses a single shared block. Larger models increase `hidden_dim`, head counts, block counts, and training loop budgets. Effective depth is tunable at inference independently of parameter count.

---

## Evaluation Metrics

Cognexis evaluation reports both **quality** and **compute** together.

### Quality Metrics

- Perplexity and negative log-likelihood
- Exact match and multiple-choice accuracy
- Pass@k for code generation
- Task-specific generation metrics (BLEU, ROUGE, rubric scores)

### Compute Metrics

- Loops executed per request (mean, median, p90, p99)
- Prefill and decode latency
- Approximate FLOPs by stage
- Peak memory and KV cache memory
- Scheduler overhead

### Depth-Aware Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **DEI** (Depth Efficiency Index) | `(M(N₂) - M(N₁)) / (C(N₂) - C(N₁))` | Quality gain per extra unit of compute. Higher = better returns from additional loops. |
| **LSP** (Loop Saturation Point) | `argmax_N DEI(N)` | Depth where marginal returns peak. Preferred default loop budget. |
| **OT** (Overthinking Threshold) | `min { d : M(d+1) < M(d) - tolerance }` | First depth where extra loops harm quality. Scheduler must stop before this. |
| **DGR** (Depth Gain Ratio) | `(M(N_max) - M(N_min)) / |M(N_min)|` | Total improvement from shallow to maximum depth. |

DEI, LSP, OT, and DGR are task-dependent — different tasks saturate and overthink at different depths.

---

## Quick Start

### Tokenize

```rust
use cognexis::tokenizer::Tokenizer;

let tokenizer = Tokenizer::from_artifact("tokenizer.json")?;
let ids = tokenizer.encode("The recurrent core", EncodeOptions::default())?;
let text = tokenizer.decode(&ids, DecodeOptions::default())?;
```

### Generate (Fixed Loop Mode)

```rust
use cognexis::{CognexisModel, GenerationRequest, LoopMode, SamplingOptions};

let model = CognexisModel::from_checkpoint("checkpoint/", &ServeConfig::load("serve.yaml")?)?;

let request = GenerationRequest {
    input_ids: tokenizer.encode("Explain transformers", EncodeOptions::default())?,
    max_new_tokens: 128,
    loop_options: LoopOptions {
        mode: LoopMode::Fixed(8),
        ..Default::default()
    },
    sampling: SamplingOptions::default()
        .with_temperature(0.7)
        .with_top_p(0.9),
};

for event in model.generate_streaming(request)? {
    println!("{}", event.text_delta);
}
```

### Generate (Adaptive Loop Mode)

```rust
let request = GenerationRequest {
    input_ids: ids,
    max_new_tokens: 128,
    loop_options: LoopOptions {
        mode: LoopMode::Adaptive {
            min_loops: 2,
            max_loops: 16,
        },
        ..Default::default()
    },
    sampling: SamplingOptions::default(),
};
// Scheduler halts when confidence is high and predicted marginal gain is low.
```

### Training

```bash
cognexis train --config configs/train.yaml
```

### Evaluation

```bash
cognexis eval --config configs/eval.yaml --checkpoint checkpoint/

# Loop scaling study
cognexis eval loop-scaling --config configs/eval.yaml --depths 1,2,4,8,12
```

---

## Project Structure

```
cognexis/
├── cognexis-spec/          # Full engineering specification (28 documents)
│   ├── spec01_overview.md
│   ├── spec02_tokenizer.md
│   ├── spec03_embedding.md
│   ├── spec04_attention.md
│   ├── spec05_feedforward.md
│   ├── spec06_transformer_block.md
│   ├── spec07_prelude.md
│   ├── spec08_recurrent_core.md
│   ├── spec09_coda.md
│   ├── spec10_lm_head.md
│   ├── spec11_config.md
│   ├── spec12_data_loading.md
│   ├── spec13_curriculum.md
│   ├── spec14_distributed_training.md
│   ├── spec15_stability_normalization.md
│   ├── spec16_prefill_decode.md
│   ├── spec17_scheduler_design.md
│   ├── spec18_tokenwise_scheduling.md
│   ├── spec19_value_head.md
│   ├── spec20_evaluation_metrics.md
│   ├── spec21_loop_scaling.md
│   ├── spec22_ablation.md
│   ├── spec23_instruction_tuning.md
│   ├── spec24_safety_monitoring.md
│   ├── spec25_glossary.md
│   ├── spec26_implementation_outline.md
│   ├── spec27_limitations_future.md
│   ├── spec28_conclusion.md
│   ├── cognexis_white_paper.md     # High-level narrative
│   └── white_paper_references.md
│
└── cognexis/               # Rust implementation
    ├── Cargo.toml
    └── src/
        tokenizer.rs        # Subword encoding/decoding, special tokens, chat templates
        embedding.rs        # Token embeddings, RoPE position IDs
        attention.rs        # MHA, GQA, MQA, sliding-window, KV cache
        feedforward.rs      # SwiGLU, GeGLU, GELU, ReLU MLPs
        transformer_block.rs# Pre-norm composition, residual scaling, gating
        prelude.rs          # Prelude stage (multiple unique blocks)
        recurrent_core.rs   # Shared recurrent block, input injection, loop execution
        coda.rs             # Coda stage (multiple unique blocks)
        lm_head.rs          # Vocabulary projection, weight tying, cross-entropy loss
        stability.rs        # RMSNorm, LayerNorm, residual scaling, spectral monitoring
        scheduler.rs        # Fixed, rule-based, value-head, hybrid schedulers
        tokenwise.rs        # Token-wise loop allocation and active masking
        value_head.rs       # Learned gain predictor for adaptive scheduling
        prefill_decode.rs   # Autoregressive inference: prefill, decode, cache management
        config.rs           # Typed YAML/JSON config with validation
        data_loading.rs     # Sharded dataset streaming, document packing, shuffling
        curriculum.rs       # Loop curriculum: warm-up, depth ramp, sampling distributions
        distributed_training.rs  # FSDP, tensor parallelism, pipeline staging
        evaluation.rs       # Evaluation harness, task traits, metric computation
        loop_scaling.rs     # Loop scaling experiment runner
        ablation.rs         # Ablation study framework and config management
        instruction_tuning.rs    # SFT data formats, chat templates, assistant-only loss masking
        safety.rs           # Input/output filtering, policy enforcement, compute budgets
        lib.rs              # Top-level exports and public API
        attention.rs        # Attention variants
        feedforward.rs      # FFN activations
        stability.rs        # Normalization
        tokenizer.rs        # Tokenization
        config.rs           # Configuration
        prefill_decode.rs   # Inference
        scheduler.rs        # Scheduling
        data_loading.rs     # Data loading
        curriculum.rs       # Loop curriculum
        distributed_training.rs  # Distributed training
        evaluation.rs       # Evaluation
        loop_scaling.rs     # Loop scaling
        ablation.rs         # Ablations
        instruction_tuning.rs    # Instruction tuning
        safety.rs           # Safety
        transformer_block.rs
        embedding.rs
        prelude.rs
        recurrent_core.rs
        coda.rs
        lm_head.rs
        tokenwise.rs
        value_head.rs
        lib.rs
```

---

## Specification

The `cognexis-spec/` directory contains the complete engineering specification for Cognexis, written as 28 focused documents. It is the authoritative implementation contract — all conforming implementations must satisfy the requirements documented there.

Documents are divided into five groups:

| Group | Documents | Coverage |
|-------|-----------|----------|
| Core Components | spec01–spec10 | Tokenizer, embedding, attention, FFN, transformer block, Prelude, recurrent core, Coda, LM head |
| System Stack | spec11–spec16 | Configuration, data loading, curriculum, distributed training, stability/normalization, prefill/decode |
| Scheduling | spec17–spec19 | Adaptive scheduler design, token-wise allocation, value head |
| Evaluation & Ops | spec20–spec24 | Evaluation metrics/protocols, loop scaling, ablation, instruction tuning, safety/monitoring |
| Project | spec25–spec28 | Glossary, implementation outline, limitations/future work, conclusion |

Key invariants every implementation must preserve:

- Token IDs are identical between training and inference for the same tokenizer and normalization settings.
- Recurrent core weights are shared across all loop iterations (one checkpoint tensor, not N copies).
- Causal masking is enforced in every attention path at every loop.
- `min_loops` and `max_loops` are enforced as hard bounds regardless of scheduler mode.
- Evaluation reports quality and compute together; quality alone is incomplete.
- Safety policies wrap generation regardless of loop mode.

---

## Safety and Observability

**Cognexis inherits all standard LLM safety concerns and adds recurrent-specific risks.**

### Required Safety Controls

- Input policy checks and prompt injection detection before prefill.
- Output policy checks after generation.
- Special-token injection prevention.
- Hard compute budget limits: max prompt tokens, max generated tokens, max loops, wall-clock timeout.
- Safety risk signals in scheduler halt decisions.
- Safety metrics measured across loop counts (depth can affect refusal accuracy).

### Required Observability

Structured logs must include (without raw prompt/output text by default):

- Request ID, checkpoint ID, tokenizer checksum.
- Prompt and generation token counts.
- Loop mode, loop count distribution, halt reasons.
- Prefill and decode latency.
- Safety filter actions, error/stop reason.

Exposed metrics: requests/sec, tokens/sec, mean loops, loop histogram, halt reason counts, refusal rate, budget exhaustion rate, non-finite activation count, error rate by category.

Tracing spans: tokenization, prefill, each recurrent loop, scheduler observation, Coda, LM head, sampling, safety filtering, streaming emission.

---

## Known Limitations

See `cognexis-spec/spec27_limitations_future.md` for the full list. Key items:

- **Stability at high loop counts.** Repeated application can explode norms, collapse to uninformative fixed points, or oscillate. Requires residual scaling, gating, spectral monitoring, and gradient clipping.
- **Latency variability.** Adaptive scheduling produces variable response times, which complicates SLA-driven serving.
- **Scheduler generalization.** Learned schedulers may halt incorrectly on out-of-distribution prompts. Conservative fallback policies are required.
- **Overthinking.** Quality can degrade beyond a threshold depth. The Overthinking Threshold varies by task and must be measured empirically.
- **KV cache subtlety.** Recurrent hidden states change with loop index, making cache reuse more error-prone than standard transformers. Correctness-first recomputation may be necessary.
- **Token-wise efficiency.** Masked dense token-wise execution is simple and correct but saves little compute. Hardware-appropriate sparse kernels are future work.
- **Training complexity.** Loop curricula, recurrent gradient handling, and scheduler training add overhead not present in standard decoder-only training.

Cognexis does not make deeper reasoning free. It makes compute cost *explicit, measurable, and controllable*. Implementations must preserve stability, report depth efficiency honestly, and enforce runtime budgets and safety policies.

---

## License

See `LICENSE` file.