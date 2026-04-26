use cognexis::ablation::{plan_ablation, AblationType};
use cognexis::attention::MultiHeadAttention;
use cognexis::config::ModelConfig;
use cognexis::evaluation::perplexity;
use cognexis::instruction_tuning::{summarize_examples, InstructionExample};
use cognexis::prefill_decode::{decode, prefill};
use cognexis::recurrent_core::RecurrentCore;
use cognexis::scheduler::{
    compute_loops_bounded, HaltReason, LoopAction, LoopScheduling, RuleBasedScheduler,
    SchedulerObservation,
};
use cognexis::stability::{estimate_spectral_norm, spectral_normalize};
use cognexis::tokenizer::{DecodeOptions, EncodeOptions, Tokenizer, TruncationPolicy};

fn tiny_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 300,
        hidden_size: 4,
        num_prelude_layers: 0,
        num_recurrent_blocks: 1,
        min_loop_count: 1,
        max_loop_count: 2,
        num_coda_layers: 0,
        num_attention_heads: 2,
        num_kv_heads: 1,
        ff_inner_dim: 8,
        norm_epsilon: 1.0e-5,
        recurrent_residual_scale: 0.25,
        tie_embeddings: true,
        embedding_scale: 1.0,
        tokenizer_path: None,
    }
}

#[test]
fn config_validation_enforces_attention_and_loop_invariants() {
    let mut config = tiny_config();
    assert!(config.validate().is_ok());

    config.hidden_size = 5;
    assert!(config.validate().is_err());

    config = tiny_config();
    config.max_loop_count = 0;
    assert!(config.validate().is_err());
}

#[test]
fn tokenizer_round_trips_unicode_and_rejects_untrusted_specials() {
    let tokenizer = Tokenizer::new();
    let text = "fn main() {\n    println!(\"hi Σ\");\n}";

    let encoded = tokenizer
        .encode_with_options(
            text,
            EncodeOptions {
                add_bos: true,
                add_eos: true,
                ..EncodeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(encoded.first(), Some(&tokenizer.bos_id()));
    assert_eq!(encoded.last(), Some(&tokenizer.eos_id()));

    let decoded = tokenizer
        .decode_with_options(
            &encoded,
            DecodeOptions {
                stop_at_eos: true,
                ..DecodeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(decoded, text);

    assert!(tokenizer
        .encode_with_options("<|system|>ignored", EncodeOptions::default())
        .is_err());

    let trusted = tokenizer
        .encode_with_options(
            "<|system|>",
            EncodeOptions {
                allow_special: true,
                ..EncodeOptions::default()
            },
        )
        .unwrap();
    assert_eq!(trusted.len(), 1);
}

#[test]
fn tokenizer_applies_explicit_truncation_policy() {
    let tokenizer = Tokenizer::new();
    let encoded = tokenizer
        .encode_with_options(
            "abcdef",
            EncodeOptions {
                max_len: Some(3),
                truncation: TruncationPolicy::Left,
                ..EncodeOptions::default()
            },
        )
        .unwrap();

    assert_eq!(tokenizer.decode(&encoded), "def");
}

#[test]
fn attention_is_causal_for_prefill_inputs() {
    let mut config = tiny_config();
    config.hidden_size = 2;
    config.num_attention_heads = 1;
    config.num_kv_heads = 1;
    let attention = MultiHeadAttention::new(&config);

    let q = vec![vec![10.0, 0.0], vec![0.0, 1.0]];
    let k = vec![vec![0.0, 1.0], vec![10.0, 0.0]];
    let v = vec![vec![1.0, 1.0], vec![100.0, 100.0]];
    let output = attention.try_forward(&q, &k, &v).unwrap();

    assert_eq!(output[0], v[0]);
    assert!(output[1][0] > 1.0);
}

#[test]
fn recurrent_core_clamps_requested_loop_count() {
    let mut config = tiny_config();
    config.min_loop_count = 2;
    config.max_loop_count = 3;
    let core = RecurrentCore::new(&config);
    let input = vec![vec![0.1, 0.2, 0.3, 0.4]];

    let below_min = core.forward(&input, 0);
    let at_min = core.forward(&input, 2);
    let above_max = core.forward(&input, 9);
    let at_max = core.forward(&input, 3);

    assert_eq!(below_min, at_min);
    assert_eq!(above_max, at_max);
    assert_ne!(at_min, input);
}

#[test]
fn scheduler_enforces_hard_bounds_before_soft_rules() {
    let scheduler = RuleBasedScheduler::default();

    let safety = scheduler.decide(SchedulerObservation {
        loops_executed: 0,
        min_loops: 2,
        max_loops: 8,
        hidden_delta: None,
        confidence: None,
        predicted_gain: None,
        remaining_loop_budget: Some(8),
        safety_halt: true,
    });
    assert_eq!(safety.action, LoopAction::Halt);
    assert_eq!(safety.halt_reason, Some(HaltReason::Safety));

    let min = scheduler.decide(SchedulerObservation {
        safety_halt: false,
        loops_executed: 1,
        min_loops: 2,
        max_loops: 8,
        hidden_delta: Some(0.0),
        confidence: Some(1.0),
        predicted_gain: Some(0.0),
        remaining_loop_budget: Some(8),
    });
    assert_eq!(min.action, LoopAction::Continue);

    let bounded = compute_loops_bounded(LoopScheduling::Fixed(12), 100, 2, 8, Some(5));
    assert_eq!(bounded, 5);
}

#[test]
fn prefill_and_decode_update_cache_shapes() {
    let config = tiny_config();
    let (hidden, mut cache) = prefill(&config, &[10, 11, 12]);
    assert_eq!(hidden.len(), 3);
    assert_eq!(hidden[0].len(), config.hidden_size);
    assert_eq!(cache.keys.len(), 3);

    let next = decode(&config, &hidden, 13, &mut cache);
    assert_eq!(next.len(), config.hidden_size);
    assert_eq!(cache.keys.len(), 4);
}

#[test]
fn perplexity_uses_stable_cross_entropy() {
    let ppl = perplexity(&[vec![10.0, 0.0], vec![0.0, 10.0]], &[0, 1]);
    assert!(ppl >= 1.0);
    assert!(ppl < 1.001);
}

#[test]
fn spectral_normalization_caps_estimated_norm() {
    let weight = vec![vec![3.0, 0.0], vec![0.0, 4.0]];
    let normalized = spectral_normalize(&weight);
    let sigma = estimate_spectral_norm(&normalized, 30);

    assert!(sigma <= 1.01);
}

#[test]
fn ablation_plan_disables_only_selected_component() {
    let plan = plan_ablation(AblationType::RemoveRecurrent);

    assert!(plan.use_attention);
    assert!(plan.use_feedforward);
    assert!(!plan.use_recurrent);
    assert!(plan.use_normalization);
}

#[test]
fn instruction_examples_are_summarized_deterministically() {
    let examples = vec![
        InstructionExample {
            prompt: "hi".to_string(),
            response: "hello".to_string(),
        },
        InstructionExample {
            prompt: "Σ".to_string(),
            response: "ok".to_string(),
        },
    ];

    let stats = summarize_examples(&examples);
    assert_eq!(stats.examples, 2);
    assert_eq!(stats.total_prompt_chars, 3);
    assert_eq!(stats.total_response_chars, 7);
}
