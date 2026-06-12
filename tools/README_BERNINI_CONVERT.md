# Bernini weight conversion (ComfyUI)

Split the joint [ByteDance/Bernini-Diffusers](https://huggingface.co/ByteDance/Bernini-Diffusers) `bernini/` checkpoint into components for ComfyUI and full-Bernini planner nodes.

## Prerequisites

```bash
pip install safetensors torch
```

Download the Bernini shard bundle:

```bash
pip install -U huggingface_hub
hf download ByteDance/Bernini-Diffusers \
  --local-dir /data/Bernini-Diffusers \
  --include "bernini/*"
```

You still need `vae/`, `t5_text_encoder/` from the same repo for full inference.

The joint `bernini/` shard also contains ~243 `t5_text_encoder.*` keys (UMT5-XXL bundled for official diffusers). The converter **skips these on purpose** — ComfyUI loads UMT5 via `CLIPLoader` type `wan` from the separate `t5_text_encoder/` folder. `mllm.*` is extracted to `bernini_mllm.safetensors` (or use HF `mllm/` directly).

## Usage

From the ComfyUI repo root:

```bash
# Classify keys only (no disk read of 180GB shards)
python tools/convert_bernini_weights.py \
  --input-dir /data/Bernini-Diffusers/bernini \
  --dry-run

# Full conversion (needs ~64GB+ RAM — accumulates one bucket at a time)
python tools/convert_bernini_weights.py \
  --input-dir /data/Bernini-Diffusers/bernini \
  --output-dir /data/bernini-comfy \
  --dtype bfloat16
```

### Low RAM (~50GB) — one bucket per run

The joint checkpoint is ~180GB. Convert **one output file at a time** with `--only`:

```bash
OUT=/data/bernini-comfy
IN=/data/Bernini-Diffusers/bernini

python tools/convert_bernini_weights.py --input-dir "$IN" --output-dir "$OUT" --only planner
python tools/convert_bernini_weights.py --input-dir "$IN" --output-dir "$OUT" --only vit_decoder
# Skip mllm if you use HF mllm/ folder directly with BerniniMLLMLoader
python tools/convert_bernini_weights.py --input-dir "$IN" --output-dir "$OUT" --only mllm --dtype bfloat16
# DiT experts: float16 halves file size (~32GB peak vs ~64GB bf16)
python tools/convert_bernini_weights.py --input-dir "$IN" --output-dir "$OUT" --only high_noise --dtype float16
python tools/convert_bernini_weights.py --input-dir "$IN" --output-dir "$OUT" --only low_noise --dtype float16
```

Peak RAM ≈ size of one output bucket (+ small shard batch). Use `--dtype none` to avoid cast copies when shard dtype already matches.

### Options

| Flag | Description |
|------|-------------|
| `--input-dir` | Folder with `model.safetensors.index.json` and `model-*-of-*.safetensors` |
| `--output-dir` | Default: `<input-dir>/comfy_split` |
| `--dtype` | `bfloat16` (default), `float16`, `float32`, `none` |
| `--comfy-prefix` | Optional prefix on DiT keys, e.g. `model.diffusion_model.` |
| `--only` | `planner`, `vit_decoder`, `mllm`, `high_noise`, `low_noise` — single-bucket mode for low RAM |
| `--dry-run` | Parse index only |
| `-v` | Verbose logging |

## Outputs

| File | Source keys | Use |
|------|-------------|-----|
| `bernini_planner.safetensors` | `connector.*`, `mask_tokens` | Planner loader (upcoming nodes) |
| `bernini_vit_decoder.safetensors` | `vit_decoder.*` | VIT flow decoder loader |
| `bernini_mllm.safetensors` | `mllm.*` (prefix stripped) | `models/bernini/` + `BerniniMLLMLoader` |
| `bernini_high_noise.safetensors` | `diff_dec.transformer.*` → Comfy keys | `UNETLoader` high-noise expert |
| `bernini_low_noise.safetensors` | `diff_dec_low.transformer_2.*` → Comfy keys | `UNETLoader` low-noise expert |

DiT outputs use ComfyUI `WanModel` key layout (`blocks.*.self_attn.q`, `head.modulation`, etc.) compatible with `UNETLoader` and Wan 2.2 blueprints.

## Where to put converted files in ComfyUI

```text
ComfyUI/models/
  bernini/
    bernini_planner.safetensors
    bernini_vit_decoder.safetensors
    bernini_mllm.safetensors          # MLLM weights (from convert script)
    mllm_processor/                   # tokenizer + preprocessor only (~few MB)
      config.json
      preprocessor_config.json
      tokenizer.json
      …
  diffusion_models/
    bernini_high_noise.safetensors
    bernini_low_noise.safetensors
  text_encoders/
    umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/
    wan_2.1_vae.safetensors
```

Processor bundle (no weight shards — **not** included in `bernini_mllm.safetensors`):

```bash
# Option A: symlink your existing Bernini-Diffusers mllm/ folder
ln -s /workspace/data/Bernini-Diffusers/mllm ComfyUI/models/bernini/mllm

# Option B: copy tokenizer/preprocessor JSON only into mllm_processor/
mkdir -p ComfyUI/models/bernini/mllm_processor
hf download ByteDance/Bernini-Diffusers \
  --local-dir /tmp/bernini-mllm-meta \
  --include "mllm/config.json" "mllm/preprocessor_config.json" "mllm/tokenizer*" \
           "mllm/chat_template.json" "mllm/merges.txt" "mllm/vocab.json"
cp /tmp/bernini-mllm-meta/mllm/* ComfyUI/models/bernini/mllm_processor/
```

If neither local folder exists, ComfyUI auto-downloads processor metadata from `ByteDance/Bernini-Diffusers` (subfolder `mllm`) on first run.

`BerniniMLLMLoader` picks `bernini_mllm.safetensors` from the dropdown; optional `hf_folder` overrides with a full HF `mllm/` directory.

## Verify UNET load (after conversion)

In ComfyUI, load `bernini_high_noise.safetensors` and `bernini_low_noise.safetensors` as separate UNET models. Check the log for missing keys; a small number of optional norms may be absent (identity layers).

## Key mapping reference

Implemented in `tools/bernini_keymaps.py`:

- Diffusers `attn1` / `attn2` → Comfy `self_attn` / `cross_attn`
- `to_q` → `q`, `to_out.0` → `o`, etc.
- `condition_embedder.*` → `text_embedding.*`, `time_embedding.*`, `time_projection.*`
- `proj_out` → `head.head`, `scale_shift_table` → `head.modulation`
- Block `scale_shift_table` → `modulation`
- Diffusers `norm2` (cross-attn) → Comfy `norm3`

## Related docs

- `docs/FULL_BERNINI_IMPLEMENTATION_PLAN.md` — full integration plan
- `Bernini/docs/ARCHITECTURE_REFERENCE.md` — official architecture reference
