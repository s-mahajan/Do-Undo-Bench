# Do-Undo Code

Run BAGEL inference on Do-Undo-Bench annotations and evaluate the generated forward/reverse images.

## Expected Layout

```text
parent_dir/
├── do-undo-bench/
│   ├── annotations_train.json
│   └── annotations_test.json
├── do-undo-code/
│   ├── inference.py
│   ├── eval.py
│   └── lavila/utils/utils.py
├── BAGEL/                 # cloned separately
└── LaViLA/                # cloned separately, needed for action classification
```

The benchmark JSON files live at `../do-undo-bench/` when commands are run from this repo.

## Data Requirements

Do-Undo-Bench[https://huggingface.co/datasets/doundo/doundobench] contains annotations only. You also need the EPIC-KITCHENS RGB frames arranged like:

```text
<EPIC_FRAMES_ROOT>/<participant_id>/rgb_frames/<video_id>/frame_0000001810.jpg
```

For example, the first test row maps to:

```text
<EPIC_FRAMES_ROOT>/P03/rgb_frames/P03_22/frame_0000001810.jpg
```

## Setup

```bash
# From the parent directory of this repo
git clone https://github.com/bytedance-seed/BAGEL.git
git clone https://github.com/facebookresearch/LaViLA.git
```

Install BAGEL and LaViLA dependencies following their upstream instructions. The scripts also use `torch`, `torchvision`, `pandas`, `numpy`, `Pillow`, `transformers`, `open_clip_torch`, `scipy`, `opencv-python`, and `tqdm`.

Set paths used by the commands:

```bash
cd do-undo-code

export BAGEL_REPO=/path/to/BAGEL
export BAGEL_MODEL=/path/to/BAGEL-7B-MoT
export LAVILA_REPO=/path/to/LaViLA
export ACTION_CKPT=/path/to/clip_openai_timesformer_large.ft_ek100_cls.ep_0090.md5sum_4a2509.pth
export EPIC_FRAMES_ROOT=/path/to/EPIC-KITCHENS/rgb_frames_root
```

`lavila/utils/utils.py` in this repo contains the DINO, CLIP, and optical-flow metric helpers. `eval.py` can load it locally, and `--lavila_repo` is required when running `--action_classify` so the LaViLA model code is importable.

## Run Inference

Generate forward and reverse BAGEL edits for the test split:

```bash
python inference.py \
  --bagel_repo "$BAGEL_REPO" \
  --model_path "$BAGEL_MODEL" \
  --pretrained_path "$BAGEL_MODEL" \
  --input_json ../do-undo-bench/annotations_test.json \
  --image_root "$EPIC_FRAMES_ROOT" \
  --output_dir outputs/bagel_test \
  --skip_errors
```

Outputs are named by annotation id:

```text
outputs/bagel_test/<narration_id>_fwd.png
outputs/bagel_test/<narration_id>_rev.png
```

Useful inference options:

- `--forward_only` runs only start-frame to forward-action edits.
- `--reverse_only` runs only end-frame to reverse-action edits.
- `--num_timesteps`, `--cfg_text_scale`, and `--cfg_img_scale` control BAGEL sampling.
- `--input_json ../do-undo-bench/annotations_train.json` runs the train split instead.

## Evaluate Outputs

Run all metrics:

```bash
python eval.py \
  --lavila_repo "$LAVILA_REPO" \
  --input_json ../do-undo-bench/annotations_test.json \
  --image_root "$EPIC_FRAMES_ROOT" \
  --output_dir outputs/bagel_test \
  --output_csv results/bagel_test_metrics.csv \
  --checkpoint_path "$ACTION_CKPT" \
  --action_classify \
  --dino \
  --clip \
  --epe
```

Run only lightweight embedding metrics:

```bash
python eval.py \
  --input_json ../do-undo-bench/annotations_test.json \
  --image_root "$EPIC_FRAMES_ROOT" \
  --output_dir outputs/bagel_test \
  --output_csv results/bagel_test_embedding_metrics.csv \
  --dino \
  --clip
```

Evaluation expects the same generated filenames produced by `inference.py`: `<narration_id>_fwd.png` and `<narration_id>_rev.png`.

## JSON Schema Used

Both scripts read Do-Undo-Bench JSON rows with these fields:

- `narration_id`, `participant_id`, `video_id`
- `start_frame`, `stop_frame`
- `narration`, `verb_class`, `noun_class`
- `forward_prompt`, `reverse_prompt`

Older CSV-style inputs are still partially supported through deprecated `--csv_path` / `--input_csv` aliases, but the recommended input is `../do-undo-bench/annotations_test.json` or `../do-undo-bench/annotations_train.json`.

