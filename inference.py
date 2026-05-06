"""
BAGEL-7B-MoT Batch Image Editing Inference Script
--------------------------------------------------
Reads a Do-Undo-Bench JSON file with columns:
  participant_id          - EPIC-KITCHENS participant identifier
  video_id                - EPIC-KITCHENS video identifier
  start_frame             - source frame index
  stop_frame              - target frame index
  forward_prompt          - prompt to edit start_img → end state
  reverse_prompt          - prompt to edit end_img   → start state

For each row, two edited images are saved:
  <output_dir>/<narration_id>_fwd.png
  <output_dir>/<narration_id>_rev.png

Usage:
  python bagel_batch_edit.py \
      --model_path /path/to/BAGEL-7B-MoT \
      --pretrained_path /path/to/BAGEL-7B-MoT \
      --input_json ../do-undo-bench/annotations_test.json \
      --image_root /path/to/EPIC-KITCHENS \
      --output_dir ./edited_outputs \
      [--cfg_text_scale 4.0] \
      [--cfg_img_scale  2.0] \
      [--num_timesteps  50]  \
      [--image_size     512] \
      [--forward_only]       \
      [--reverse_only]       \
      [--skip_errors]
"""

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
# ─── Model loading ────────────────────────────────────────────────────────────


def add_repo_to_path(repo_path: str | None):
    if repo_path:
        sys.path.insert(0, str(Path(repo_path).expanduser().resolve()))

def load_model(model_path: str, pretrained_path: str):
    """
    Load BAGEL-7B-MoT using the official accelerate-based pattern from inference.ipynb.
    Must be run from the Bagel repo root so that modeling/ and data/ are importable.
    """
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
        from transformers import AutoTokenizer
        from data.data_utils import add_special_tokens
        from modeling.bagel import BagelConfig, Bagel
        from modeling.bagel import Qwen2Config
        from modeling.bagel.siglip_navit import SiglipVisionConfig
        from modeling.bagel.siglip_navit import SiglipVisionModel
        from modeling.bagel import Qwen2ForCausalLM
        from modeling.bagel.siglip_navit import SiglipVisionModel
        from modeling.autoencoder import load_ae
        from data.data_utils import add_special_tokens
        from modeling.qwen2 import Qwen2Tokenizer
    except ImportError as e:
        sys.exit(
            f"[ERROR] Could not import BAGEL modules: {e}\n"
            "Run with --bagel_repo /path/to/BAGEL or from the BAGEL repo root."
        )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("[INFO] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # ── VAE ───────────────────────────────────────────────────────────────────
    print("[INFO] Loading VAE ...")
    # Patch load_sft to load on CPU first — avoids OSError 19 on Linux/NFS
    # systems where safetensors cannot map tensors directly onto a CUDA device.
    import modeling.autoencoder as _ae_mod
    import safetensors.torch as _sft
    _orig_load_sft = _ae_mod.load_sft

    def _cpu_load(path, device="cpu"):
        # safe_open / load_file use mmap internally, which fails on NFS/network
        # volumes (OSError 19). safetensors.torch.load(bytes) bypasses mmap entirely.
        path = os.path.join(path, "ae.safetensors")
        print(f"[INFO] Loading VAE weights (no-mmap) from {path!r}")
        with open(path, "rb") as fh:
            data = fh.read()
        return _sft.load(data)   # load(bytes) — no mmap, returns {str: Tensor}
    _ae_mod.load_sft = _cpu_load
    try:
        vae_model, vae_config = load_ae(pretrained_path)
    finally:
        _ae_mod.load_sft = _orig_load_sft  # restore original


    # ── Configs ───────────────────────────────────────────────────────────────
    llm_config = Qwen2Config.from_json_file(os.path.join(pretrained_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(pretrained_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    # ── Model (empty weights → dispatch) ─────────────────────────────────────
    print("[INFO] Instantiating BAGEL model ...")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    n_gpus = torch.cuda.device_count()
    
    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    print("[INFO] Model ready.")
    return model, tokenizer, vae_model, vae_config, new_token_ids


# ─── Single edit ──────────────────────────────────────────────────────────────

def run_edit(
    inferencer,
    image: Image.Image,
    prompt: str,
    cfg_text_scale: float,
    cfg_img_scale: float,
    num_timesteps: int,
    image_size: int,
) -> Image.Image:
    """Run one image-editing pass and return a PIL Image."""
    show_thinking=False
    image = image.convert("RGB")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_hyper = dict(
            max_think_token_n=max_think_token_n if show_thinking else 1024,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            timestep_shift=3.0,
            num_timesteps=num_timesteps,
            # cfg_renorm_min=cfg_renorm_min,
            # cfg_renorm_type=cfg_renorm_type,
        )
        result = inferencer(
        image=image, text=prompt, think=show_thinking, **inference_hyper)
    

    if isinstance(result, dict):
        out_img = result.get("image") or result.get("images")
        if isinstance(out_img, list):
            out_img = out_img[0]
    elif isinstance(result, Image.Image):
        out_img = result
    else:
        raise ValueError(f"Unexpected inferencer output type: {type(result)}")

    return out_img


def parse_prompt(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except (SyntaxError, ValueError):
            pass
    return value


def load_rows(input_path: Path):
    if input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Expected {input_path} to contain a JSON array")
        return rows

    with open(input_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def frame_path(row, image_root: Path, frame_key: str) -> str:
    frame_id = int(row[frame_key])
    return str(
        image_root
        / row["participant_id"]
        / "rgb_frames"
        / row["video_id"]
        / f"frame_{frame_id:010d}.jpg"
    )


def sample_id(row, fallback_path: str) -> str:
    return row.get("narration_id") or Path(fallback_path).stem


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BAGEL-7B batch image editing from Do-Undo-Bench JSON")
    parser.add_argument("--bagel_repo",      type=str, default=None,
                        help="BAGEL repo root. Needed unless BAGEL modules are already importable.")
    parser.add_argument("--model_path",      type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_json", "--json_path", dest="input_path", type=Path,
                             help="Do-Undo-Bench JSON annotation file")
    input_group.add_argument("--csv_path", dest="legacy_csv_path", type=Path,
                             help="Deprecated alias for older CSV inputs")
    parser.add_argument("--image_root",      type=str, required=True)
    parser.add_argument("--output_dir",      default="./edited_outputs")
    parser.add_argument("--cfg_text_scale",  type=float, default=4.0)
    parser.add_argument("--cfg_img_scale",   type=float, default=2.0)
    parser.add_argument("--num_timesteps",   type=int,   default=50)
    parser.add_argument("--image_size",      type=int,   default=512)
    parser.add_argument("--skip_errors",     action="store_true",
                        help="Log errors and continue instead of crashing")
    parser.add_argument("--forward_only",   action="store_true",
                        help="Only run forward edits (start_img + forward_prompt)")
    parser.add_argument("--reverse_only",   action="store_true",
                        help="Only run reverse edits (end_img + reverse_prompt)")
    args = parser.parse_args()

    add_repo_to_path(args.bagel_repo)
    try:
        from data.transforms import ImageTransform
    except ImportError as e:
        sys.exit(
            f"[ERROR] Could not import BAGEL ImageTransform: {e}\n"
            "Run with --bagel_repo /path/to/BAGEL or from the BAGEL repo root."
        )

    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer, vae_model, vae_config, new_token_ids = load_model(
        args.model_path,
        args.pretrained_path
    )

    # Build inferencer once — reused across all rows
    try:
        from inferencer import InterleaveInferencer
    except ImportError:
        sys.exit("[ERROR] Could not import 'inferencer'. Run from the Bagel repo root.")

        
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    # ── Read annotations ─────────────────────────────────────────────────────
    input_path = args.input_path or args.legacy_csv_path
    rows = load_rows(input_path)
    print(f"[INFO] {len(rows)} rows found in {input_path}")

    run_fwd = not args.reverse_only
    run_rev = not args.forward_only

    edit_kwargs = dict(
        inferencer=inferencer,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        num_timesteps=args.num_timesteps,
        image_size=args.image_size,
    )

    success, failed = 0, 0

    for idx, row in enumerate(rows):
        start_img_path = row.get("start_img") or row.get("final_start_img") or frame_path(row, image_root, "start_frame")
        end_img_path = row.get("end_img") or row.get("final_end_img") or frame_path(row, image_root, "stop_frame")
        fwd_prompt = parse_prompt(row.get("forward_prompt") or row.get("forward_prompt_qwen30b") or "")
        rev_prompt = parse_prompt(row.get("reverse_prompt") or row.get("reverse_prompt_qwen30b") or "")
        row_id = sample_id(row, start_img_path)


        # Validate required fields for chosen directions
        missing = []
        if run_fwd and not start_img_path: missing.append("start_img")
        if run_fwd and not fwd_prompt:     missing.append("forward_prompt")
        if run_rev and not end_img_path:   missing.append("end_img")
        if run_rev and not rev_prompt:     missing.append("reverse_prompt")
        if missing:
            print(f"[WARN] Row {idx+1}: missing columns {missing} — skipping.")
            failed += 1
            continue

        # ── Forward edit: start_img → edited by forward_prompt ────────────────
        if run_fwd:
            fwd_out = str(output_dir / f"{row_id}_fwd.png")
            print(f"[{idx+1}/{len(rows)}] FWD  {start_img_path}")
            print(f"           prompt: {fwd_prompt}")
            try:
                img = Image.open(start_img_path).convert("RGB")
                out = run_edit(image=img, prompt=fwd_prompt, **edit_kwargs)
                out.save(fwd_out)
                print(f"           ✓ saved → {fwd_out}")
                success += 1
            except Exception as e:
                print(f"           ✗ ERROR: {e}")
                failed += 1
                if not args.skip_errors:
                    raise

        # ── Reverse edit: end_img → edited by reverse_prompt ──────────────────
        if run_rev:
            rev_out = str(output_dir / f"{row_id}_rev.png")
            print(f"[{idx+1}/{len(rows)}] REV  {end_img_path}")
            print(f"           prompt: {rev_prompt}")
            try:
                img = Image.open(end_img_path).convert("RGB")
                out = run_edit(image=img, prompt=rev_prompt, **edit_kwargs)
                out.save(rev_out)
                print(f"           ✓ saved → {rev_out}")
                success += 1
            except Exception as e:
                print(f"           ✗ ERROR: {e}")
                failed += 1
                if not args.skip_errors:
                    raise

    print(f"\n[DONE] {success} edits succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
