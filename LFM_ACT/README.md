# Native LFM2.5-VL-450M

A readable, trainable PyTorch implementation of LiquidAI's released checkpoint:
LFM2 hybrid language backbone, SigLIP2 NaFlex vision encoder, multimodal projector,
image/chat processing, cached generation, and a saved tiktoken tokenizer.

There are **no Transformers imports in runtime code**. Runtime dependencies are
PyTorch, torchvision, Pillow, safetensors, and tiktoken. The original weights are
loaded directly; no converted model-weight file is created.

## Quick start

Run from the repository root. `uv sync` installs the declared runtime dependencies.
The bundled tokenizer artifact is ready to load; conversion is not a startup step.

```bash
python -m LFM_ACT.generate
```

For text only, use `{"role": "user", "content": "Hello!"}`. Images can be local
paths, PIL images, or uint8 RGB tensors of shape `[3, height, width]`. Floating image
tensors are intentionally rejected rather than guessing whether they use 0–1 or
0–255 values. URLs are not fetched by the processor.

`generate` defaults to greedy decoding. Optional `temperature`, `top_p`, `min_p`,
and `repetition_penalty` control sampling. Generation supports left-padded batches
and returns prompt plus generated tokens. It uses a fresh cache per call.

## Architecture and interfaces

| File | Responsibility |
| --- | --- |
| `config.py` | Editable dataclasses and loading original/native JSON configuration |
| `language.py` | Gated causal depthwise convolutions, GQA, Q/K RMSNorm, RoPE, SwiGLU |
| `vision.py` | SigLIP2 NaFlex patch embeddings, position interpolation, transformer layers |
| `model.py` | Spatial downsampling, projector, image insertion, tied output projection, loss |
| `processor.py` | Resize/tiling/thumbnail processing, patch masks, chat formatting |
| `cache.py`, `generation.py` | Per-request attention/convolution state and autoregressive decoding |
| `load_weights.py` | Validated selective safetensors loading and explicit layer remapping |
| `training.py` | Freezing, LoRA injection, trainable parameter selection |
| `tokenizer.py`, `convert_tokenizer.py` | Saved tiktoken runtime and one-time conversion |

The default language model has 16 blocks (10 convolution, 6 attention), hidden
width 1,024, 16 query heads, and 8 KV heads. Its **effective FFN width is 4,608**;
the original config's 6,656 is adjusted by upstream SwiGLU configuration rules.
The vision model has 12 blocks, width 768, and 16×16 RGB patches. The projector
combines 2×2 patches in the checkpoint's channel order, then applies
3,072 → 2,048 → 1,024 linear layers with GELU.

Configuration defaults describe this checkpoint. `config.save(path)` writes a
plain JSON configuration; `ModelConfig.from_pretrained(directory)` reads either
that native `config.json` or the original checkpoint configuration. The initial
supported context limit is 32,768, matching the model card, despite the original
RoPE position configuration allowing 128,000. Increasing `context_length` is an
explicit experiment, not a claim of validated long-context quality.

The final projection uses `model.output_weight`, which is the same parameter as
`model.model.language_model.embed_tokens.weight`. There is no independent LM-head
weight to load or freeze.

`forward` returns a `ModelOutput` with `logits`, `last_hidden_state`,
`image_features`, `cache`, and optional `loss`. Useful entry points:

```python
features = model.encode_images(pixel_values, spatial_shapes, pixel_attention_mask)
output = model(**batch, return_logits=False)  # avoid the full vocabulary projection
hidden_states = output.last_hidden_state
output = model(**batch, logits_to_keep=1)  # only final-position logits
```

`inputs_embeds` can replace `input_ids` for custom text/action embeddings. Image
insertion through the standard multimodal interface requires `input_ids` so
placeholder positions are explicit. A supplied `ModelCache` is updated in place;
use a new cache for a new request or after changing weights/architecture. Cached
attention masks cover the complete prefix plus current input. Chunked and
single-token continuation are supported, including all-convolution backbones.

## Load unchanged parts of an edited architecture

The loader validates the complete selected mapping **before copying any tensors**.
Shape mismatches always raise `WeightLoadingError`; its `.report` identifies them.
It does not silently slice tensors, ignore mismatches, or initialize new layers.
Normal PyTorch constructors initialize new layers before loading.

Keep the first half of the language model:

```python
from dataclasses import replace

config = ModelConfig.from_pretrained(checkpoint)
config.text = replace(config.text, layer_types=config.text.layer_types[:8])
model = LFMModel(config).to(device="cuda", dtype=torch.bfloat16)
report = load_weights(model, checkpoint, allow_unused=True)
print(report.unused)  # removed checkpoint layers; review the explicit list
```

Insert a new convolution block at index 2 and shift the original later blocks:

```python
config = ModelConfig.from_pretrained(checkpoint)
original_types = config.text.layer_types
config.text = replace(
    config.text, layer_types=original_types[:2] + ("conv",) + original_types[2:]
)
model = LFMModel(config).to(device="cuda", dtype=torch.bfloat16)
prefix = "model.language_model.layers."
remap = {prefix + str(i): prefix + str(i + 1) for i in range(2, len(original_types))}
report = load_weights(
    model, checkpoint, prefix_map=remap, allow_missing=(prefix + "2.*",)
)
```

`prefix_map` maps **checkpoint prefixes → model prefixes**, with longest-prefix
matching. Mapping a source prefix to `None` explicitly omits it. `include` selects
**target** subtrees, e.g. `include=("model.vision_tower",)`. `allow_missing` accepts
shell-style target parameter patterns. `allow_unused=True` permits checkpoint
parameters with no selected destination, which are still reported. A report lists
loaded, missing, unused, skipped, and incompatible entries. Missing entries allowed
by an explicit pattern retain their initialization.

Load before injecting LoRA and before creating the optimizer. Instantiate on a real
CPU/CUDA device; this loader does not materialize meta-device models or dispatch
across devices. The shipped loader handles the single-file released checkpoint.

## Full-parameter, frozen, and LoRA training

The model has ordinary PyTorch gradients. It does not include a dataset, ACT action
head, or training framework. The following configurations compose with your own
training loop.

```python
from LFM_ACT.training import set_trainable, inject_lora, trainable_parameters

# Full-parameter training:
set_trainable(model, True)

# Alternatively, freeze everything and train a new/replaced layer:
set_trainable(model, False)
set_trainable(model.model.language_model.layers[2], True)

# Optionally adapt selected pretrained attention projections with LoRA:
inject_lora(
    model.model.language_model, target_names=("q_proj", "v_proj"), rank=8, alpha=16
)

optimizer = torch.optim.AdamW(trainable_parameters(model).values(), lr=1e-4)
model.train()
labels = batch["input_ids"].clone()
# For supervised chat training, set prompt/system/user label positions to -100.
# Use add_generation_prompt=False and include the assistant answer in the batch.
output = model(**batch, labels=labels)
output.loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

`inject_lora` freezes each adapted base Linear but leaves other parameters' existing
trainability unchanged. Freeze the whole model first for adapter-only training;
then selectively unfreeze newly added modules. Inject into the vision or projector
subtree in the same way, selecting the desired Linear names. LoRA B starts at zero,
so injection preserves initial outputs. `LoRALinear.merged_linear()` returns a new
Linear containing the adapter update; evaluate with dropout disabled when checking
merged equivalence.

Loss is shifted next-token cross entropy. Padding and `<image>` placeholder targets
are excluded automatically; **assistant-only supervision is not inferred**. Pass
labels with `-100` at any other unsupervised positions. Do not reuse inference caches
across optimizer updates. Full-parameter BF16 training should use an appropriate
optimizer/precision policy for your workload; FP32 parameters with BF16 autocast
remain supported.

## Saved tiktoken tokenizer

`tokenizer/vocab.tiktoken` stores byte tokens with internal BPE ranks.
`tokenizer/config.json` stores the regex, special/ordinary added-token handling,
BOS/EOS/PAD IDs, original-token-ID translation, and source hash. These files form an
independent tokenizer artifact; keep both when copying it. There is no runtime
conversion or dependence on the original `tokenizer.json`.

Internal ranks are distinct from model-facing IDs. This matters for repeated
newlines: the released tokenizer's IDs do not follow merge priority. `Mathias` and
`python` are ordinary added tokens and must match literally without being removed
by `skip_special_tokens=True`. Encoding preserves all released token IDs. The
64,400-entry tokenizer must not be confused with the model's 65,536 output rows;
decoding an undefined model ID raises an explicit error.

To rebuild once from the original files:

```bash
uv run --with tokenizers python -m LFM_ACT.convert_tokenizer \
  /home/yuxin/workspace/data/models/LFM2.5-VL-450M LFM_ACT/tokenizer
```

Conversion requires the standalone `tokenizers` package only to validate against the
original tokenizer before writing the artifact. Runtime does not need it.

`encode` adds BOS by default, matching the source postprocessor. Pass
`add_special_tokens=False` for an already formatted chat prompt. Decoding defaults
to the original whitespace cleanup setting; use
`clean_up_tokenization_spaces=False` for literal byte-level round trips.

## Validation

Fast native tests use small CPU models and cover cached/chunked/padded inference,
multimodal gradients, truncated/inserted layers, atomic rejection of incompatible
loads, selective loading, frozen weights, LoRA updates, and greedy generation:

```bash
uv run --with pytest python -m pytest LFM_ACT/tests/test_native.py -q
```

Optional reference tests use **Transformers 5.1.0 only as a validation dependency**.
They check tokenizer IDs/decoding, chat templates, image preprocessing, actual
checkpoint features/logits, padded batches, and greedy generation. They require the
original checkpoint and enough memory for two models:

```bash
uv run --with 'transformers==5.1.0' python -m LFM_ACT.tests.validate_reference \
  /home/yuxin/workspace/data/models/LFM2.5-VL-450M --dtype float32
```

Run again with `--dtype bfloat16` for the released weight precision. Numerical
agreement tolerances account for attention kernels and floating-point arithmetic;
bitwise equality is not assumed across devices. Tokenizer benchmarks are measured
separately from prompt formatting, image processing, model loading, and generation.
A tiktoken backend is not a universal speedup over the Rust `tokenizers` backend.

### Recorded results

Validation on the available RTX 4070 Ti with PyTorch 2.14.1 and Transformers 5.1.0:

- 10 native CPU regression tests passed.
- All 349 checkpoint tensors loaded, with no missing or unused tensors.
- 79,559 tokenizer encoding/decoding cases matched exactly.
- Image preprocessing, vision features, and projected features matched exactly in
  the tested FP32 and BF16 cases, including tiling and multiple images.
- FP32 prefill logit errors were at most 0.000044; the tested BF16 prefill errors
  were at most 0.25 for the tiled case. Text and multimodal greedy generations matched.
- Native chunked-versus-full evaluation had maximum logit differences of 0.000103
  (FP32) and 0.75 (BF16), within the stated absolute/relative tolerances.
- A real-checkpoint BF16 multimodal backward pass produced finite, nonzero gradients
  in the vision encoder, projector, convolution, attention, and embedding weights.
  Neither Transformers nor tokenizers was imported in that runtime environment.

For one short prompt, warm encoding took about **4.45 µs** with the native tiktoken
wrapper, versus **17.94 µs** with the Transformers wrapper and **11.52 µs** with
direct tokenizers. For a batch of 1,000 longer prompts, native tiktoken took about
**46.8 ms**, versus **28.7 ms** with direct tokenizers. These are synthetic workload
measurements, not a universal speed claim. The wrapper matches literals once in
Python and performs BPE with tiktoken's Rust backend.

Machine-readable reference results are in `tests/validation_results.json`.

## Sources and attribution

The model and image-processing computations are adapted from Hugging Face
Transformers **v5.1.0**, copyright 2025 the Hugging Face team, under Apache-2.0:

- [LFM2 language model](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/lfm2/modeling_lfm2.py)
- [SigLIP2 vision model](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/siglip2/modeling_siglip2.py)
- [LFM2-VL assembly and processing](https://github.com/huggingface/transformers/tree/v5.1.0/src/transformers/models/lfm2_vl)
- [LiquidAI model and tokenizer](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), local download revision `fc6221ca597f3315e4f82fc2df606783267b34ba`

See `LICENSE.apache-2.0` for the upstream code license and `LICENSE.model` for the
released model/tokenizer license. No pretrained model weights are bundled here.
