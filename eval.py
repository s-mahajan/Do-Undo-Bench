import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

VERB_ANTONYM_MAP = {
    3: 4,    # open -> close
    4: 3,    # close -> open
    6: 8,    # turn-on -> turn off
    8: 6,
    0: 1,    # take -> put
    1: 0,    # put -> take
    11: 11,  # move and move
}


class EK100CLS(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.visual = base_model.visual
        self.fc_cls = nn.ModuleList([
            nn.Linear(1024, 97),    # verb
            nn.Linear(1024, 300),   # noun
            nn.Linear(1024, 3806),  # action
        ])

    def forward(self, x):
        feat = self.visual(x)
        return self.fc_cls[0](feat), self.fc_cls[1](feat), self.fc_cls[2](feat)


spatial_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

mean = torch.tensor([108.3272985/255, 116.7460125/255, 104.09373615/255]).view(3, 1, 1, 1)
std = torch.tensor([68.5005327/255, 66.6321579/255, 70.32316305/255]).view(3, 1, 1, 1)


def add_repo_to_path(repo_path: str | None):
    if repo_path:
        sys.path.insert(0, str(Path(repo_path).expanduser().resolve()))



def load_metric_utils():
    try:
        from lavila.utils.utils import CLIPT, DINOSCORE, generate_optical_flow
        return CLIPT, DINOSCORE, generate_optical_flow
    except ImportError:
        import importlib.util

        utils_path = Path(__file__).resolve().parent / 'lavila' / 'utils' / 'utils.py'
        spec = importlib.util.spec_from_file_location('do_undo_lavila_utils', utils_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CLIPT, module.DINOSCORE, module.generate_optical_flow


def load_and_prepare_frames(start_img_path, end_img_path):
    """Load start+end frames, interpolate to 16, apply transforms."""
    target_h, target_w = 256, 456
    frames = []
    for path in [start_img_path, end_img_path]:
        img = Image.open(path).convert('RGB')
        img = img.resize((target_w, target_h), resample=Image.BILINEAR)
        frame = torch.tensor(np.array(img), dtype=torch.float32)
        frames.append(frame)

    frames = torch.stack(frames, dim=0)
    height, width = frames.shape[1], frames.shape[2]
    frames_nchw = frames.permute(3, 0, 1, 2).unsqueeze(0)
    frames_16 = F.interpolate(
        frames_nchw,
        size=(16, height, width),
        mode='trilinear',
        align_corners=False,
    )
    frames_16 = frames_16.squeeze(0).permute(1, 2, 3, 0)

    processed = []
    for frame_idx in range(16):
        frame = frames_16[frame_idx].permute(2, 0, 1) / 255.0
        frame = spatial_transform(frame)
        processed.append(frame)

    frames_input = torch.stack(processed, dim=1)
    frames_input = (frames_input - mean) / std
    return frames_input.unsqueeze(0)


def load_action_model(checkpoint_path, device):
    from lavila.models.models import CLIP_OPENAI_TIMESFORMER_LARGE

    base = CLIP_OPENAI_TIMESFORMER_LARGE(num_frames=16, drop_path_rate=0.)
    model = EK100CLS(base)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}

    incompatible = model.load_state_dict(state_dict, strict=False)
    print("Missing:", incompatible.missing_keys)
    print("Unexpected:", incompatible.unexpected_keys)

    return model.eval().to(device)


def compute_actrecog(model, st, end, verb_gt, noun_gt, device):
    frames_input = load_and_prepare_frames(st, end).to(device)

    with torch.no_grad():
        verb_logits, noun_logits, _ = model(frames_input)

    verb_pred = verb_logits.argmax(dim=-1).item()
    noun_pred = noun_logits.argmax(dim=-1).item()
    verb_correct = int(verb_pred == verb_gt)
    noun_correct = int(noun_pred == noun_gt)
    return verb_pred, noun_pred, verb_correct, noun_correct


def build_argparser():
    parser = argparse.ArgumentParser(description='Evaluate generated Do-Undo-Bench image results')
    parser.add_argument('--lavila_repo', type=str, default=None,
                        help='LaViLA repo root. Needed unless lavila is already importable.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_json', '--json_path', dest='input_path', type=Path,
                             help='Do-Undo-Bench JSON annotation file')
    input_group.add_argument('--input_csv', dest='legacy_csv_path', type=Path,
                             help='Deprecated alias for older CSV inputs')
    parser.add_argument('--image_root', type=Path, required=True,
                        help='EPIC-KITCHENS frame root containing <participant_id>/rgb_frames/<video_id>/frame_*.jpg')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory containing generated images')
    parser.add_argument('--output_csv', type=Path, required=True, help='Path to write the evaluation CSV')
    parser.add_argument('--checkpoint_path', type=Path, default=None, help='LaViLA action classifier checkpoint path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device for action classification')
    parser.add_argument('--action_classify', action='store_true', help='Run LaViLA action classification metrics')
    parser.add_argument('--dino', action='store_true', help='Run DINO similarity metrics')
    parser.add_argument('--clip', action='store_true', help='Run CLIP text/image metrics')
    parser.add_argument('--epe', action='store_true', help='Run optical-flow EPE metrics')
    return parser


def row_value(row, key, default=None):
    if isinstance(row, dict):
        return row.get(key, default)
    return row[key] if key in row.index else default


def load_rows(input_path: Path):
    if input_path.suffix.lower() == '.json':
        with open(input_path, encoding='utf-8') as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f'Expected {input_path} to contain a JSON array')
        return rows

    with open(input_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_prompt(value):
    value = (value or '').strip()
    if value.startswith('[') and value.endswith(']'):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except (SyntaxError, ValueError):
            pass
    return value


def to_int(value):
    return int(value) if value is not None and value != '' else None


def frame_path(row, image_root: Path, frame_key: str) -> str:
    frame_id = int(row_value(row, frame_key))
    return str(
        image_root
        / row_value(row, 'participant_id')
        / 'rgb_frames'
        / row_value(row, 'video_id')
        / f'frame_{frame_id:010d}.jpg'
    )


def sample_id(row, fallback_path: str) -> str:
    return row_value(row, 'narration_id') or Path(fallback_path).stem


def generated_image_path(row, output_dir: Path, direction: str, fallback_name: str) -> str:
    generated_key = f'generated_{direction}'
    generated_name = row_value(row, generated_key)
    if generated_name:
        return str(output_dir / str(generated_name))
    return str(output_dir / fallback_name)


def print_summary(results_df):
    rverb_acc = results_df['rev_verb_correct'].mean() * 100
    rnoun_acc = results_df['rev_noun_correct'].mean() * 100
    fverb_acc = results_df['fwd_verb_correct'].mean() * 100
    fnoun_acc = results_df['fwd_noun_correct'].mean() * 100
    verb_acc = results_df['verb_correct'].mean() * 100
    noun_acc = results_df['noun_correct'].mean() * 100
    dino_f = results_df['dino_score_f'].mean()
    dino_r = results_df['dino_score_r'].mean()
    clip_f = results_df['clip_score_f'].mean()
    epe_f = results_df['epe_fwd'].mean()
    epe_r = results_df['epe_rev'].mean()
    print(f"\n Verb Accuracy: {verb_acc:.2f}%")
    print(f" Noun Accuracy: {noun_acc:.2f}%")
    print(f"\n Fwd Verb Accuracy: {fverb_acc:.2f}%")
    print(f"Fwd Noun Accuracy: {fnoun_acc:.2f}%")
    print(f"\n Rev Verb Accuracy: {rverb_acc:.2f}%")
    print(f"Rev Noun Accuracy: {rnoun_acc:.2f}%")
    print(f"\n dino fwd Accuracy: {dino_f:.2f}")
    print(f"dino rev Accuracy: {dino_r:.2f}")
    print(f"\n clip fw Accuracy: {clip_f:.2f}")
    print(f"\n epe fw Accuracy: {epe_f:.2f}")
    print(f"  spatial fidelity Accuracy: {epe_r:.2f}")


def parse_args():
    return build_argparser().parse_args()


def main():
    args = parse_args()
    if args.action_classify and args.checkpoint_path is None:
        raise ValueError('--checkpoint_path is required when --action_classify is set')

    add_repo_to_path(args.lavila_repo)
    if args.dino or args.clip or args.epe:
        CLIPT, DINOSCORE, generate_optical_flow = load_metric_utils()
    else:
        CLIPT = DINOSCORE = generate_optical_flow = None

    input_path = args.input_path or args.legacy_csv_path
    rows = load_rows(input_path)
    results = []
    device = torch.device(args.device)
    model = load_action_model(args.checkpoint_path, device) if args.action_classify else None
    dino = DINOSCORE() if args.dino else None
    clip_model = CLIPT() if args.clip else None

    for idx, row in enumerate(rows):
        action_id = row_value(row, 'narration_id')
        narration = row_value(row, 'narration')
        verb_gt = to_int(row_value(row, 'verb_class'))
        noun_gt = to_int(row_value(row, 'noun_class'))
        start_img = row_value(row, 'start_img') or row_value(row, 'final_start_img') or frame_path(row, args.image_root, 'start_frame')
        end_img = row_value(row, 'end_img') or row_value(row, 'final_end_img') or frame_path(row, args.image_root, 'stop_frame')
        row_id = sample_id(row, start_img)
        verb_rev = VERB_ANTONYM_MAP.get(verb_gt, verb_gt)
        fwd_img = generated_image_path(row, args.output_dir, 'fwd', f'{row_id}_fwd.png')
        rev_img = generated_image_path(row, args.output_dir, 'rev', f'{row_id}_rev.png')

        try:
            rev_verb_pred = row_value(row, 'rev_verb_pred')
            rev_verb_correct = row_value(row, 'rev_verb_correct')
            rev_noun_pred = row_value(row, 'rev_noun_pred')
            rev_noun_correct = row_value(row, 'rev_noun_correct')
            fwd_verb_pred = row_value(row, 'fwd_verb_pred')
            fwd_verb_correct = row_value(row, 'fwd_verb_correct')
            fwd_noun_pred = row_value(row, 'fwd_noun_pred')
            fwd_noun_correct = row_value(row, 'fwd_noun_correct')
            verb_pred = row_value(row, 'verb_pred')
            verb_correct = row_value(row, 'verb_correct')
            noun_pred = row_value(row, 'noun_pred')
            noun_correct = row_value(row, 'noun_correct')
            dino_score_f = row_value(row, 'dino_score_f')
            dino_score_r = row_value(row, 'dino_score_r')
            clip_t_f = row_value(row, 'clip_t_f', row_value(row, 'clip_score_f'))
            mean_epe = row_value(row, 'epe_fwd')
            mean_r = row_value(row, 'epe_rev')

            if args.action_classify:
                verb_pred, noun_pred, verb_correct, noun_correct = compute_actrecog(model, start_img, end_img, verb_gt, noun_gt, device)
                fwd_verb_pred, fwd_noun_pred, fwd_verb_correct, fwd_noun_correct = compute_actrecog(model, start_img, fwd_img, verb_gt, noun_gt, device)
                rev_verb_pred, rev_noun_pred, rev_verb_correct, rev_noun_correct = compute_actrecog(model, end_img, rev_img, verb_rev, noun_gt, device)

            if args.dino:
                dino_score_f = dino.score_sample(end_img, fwd_img)
                dino_score_r = dino.score_sample(start_img, rev_img)

            if args.clip:
                clip_text = parse_prompt(row_value(row, 'forward_prompt')) or narration
                clip_t_f, _ = clip_model.score_sample(end_img, fwd_img, clip_text)

            if args.epe:
                of_f_gt, _ = generate_optical_flow(start_img, end_img)
                of_f, _ = generate_optical_flow(start_img, fwd_img)
                epe_f = np.linalg.norm(of_f_gt - of_f, axis=-1)
                mean_epe = float(epe_f.mean())
                _, of_r = generate_optical_flow(start_img, rev_img)
                mean_r = float(of_r)

            results.append({
                'action_id': action_id,
                'narration': narration,
                'start_img': start_img,
                'end_img': end_img,
                'generated_fwd': fwd_img,
                'generated_rev': rev_img,
                'verb_gt': verb_gt,
                'verb_rev': verb_rev,
                'noun_gt': noun_gt,
                'rev_verb_pred': rev_verb_pred,
                'rev_verb_correct': rev_verb_correct,
                'rev_noun_pred': rev_noun_pred,
                'rev_noun_correct': rev_noun_correct,
                'fwd_verb_pred': fwd_verb_pred,
                'fwd_verb_correct': fwd_verb_correct,
                'fwd_noun_pred': fwd_noun_pred,
                'fwd_noun_correct': fwd_noun_correct,
                'verb_pred': verb_pred,
                'verb_correct': verb_correct,
                'noun_pred': noun_pred,
                'noun_correct': noun_correct,
                'dino_score_f': dino_score_f,
                'dino_score_r': dino_score_r,
                'clip_score_f': clip_t_f,
                'epe_fwd': mean_epe,
                'epe_rev': mean_r,
            })

            print(f"[{idx}] {narration} | "
                  f"revVerb: {verb_rev}→{rev_verb_pred} ({'✓' if rev_verb_correct else '✗'}) | "
                  f"revNoun: {noun_gt}→{rev_noun_pred} ({'✓' if rev_noun_correct else '✗'})  | "
                  f"fwdVerb: {verb_gt}→{fwd_verb_pred} ({'✓' if fwd_verb_correct else '✗'}) | "
                  f"fwdNoun: {noun_gt}→{fwd_noun_pred} ({'✓' if fwd_noun_correct else '✗'}) | "
                  f"fwddino: {dino_score_f} |"
                  f"revdino: {dino_score_r} |"
                  f"fwdclip: {clip_t_f} |"
                  f"mean_epe: {mean_epe} |"
                  f"mean_r: {mean_r} ")

        except Exception as e:
            print(f"[{idx}] ERROR on {action_id}: {e}")
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print_summary(results_df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
