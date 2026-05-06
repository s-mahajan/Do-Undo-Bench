
import torch
import torch.nn as nn
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large
import matplotlib.pyplot as plt
from scipy import spatial
import open_clip
from torchvision.utils import flow_to_image
from torchvision.transforms import ToPILImage
# import clip
from transformers import  CLIPModel, CLIPTokenizer, CLIPProcessor,CLIPImageProcessor
from transformers import AutoModel
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image

# import torchvision.transforms as transforms
import glob
to_pil = ToPILImage()

@torch.no_grad
class DINOSCORE:
    def __init__(
        self,
        use_gpu=True,
        size=512,
        device=None,
    ):
        super().__init__()
        self.size = size
        self.results = []
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nn = AutoModel.from_pretrained('facebook/dinov2-large').to(self.device)
        self.nn.eval()
        self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.msk_transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    
    def _load_and_preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    
    def _extract_features(self, img_path):
        img_tensor = self._load_and_preprocess(img_path)
        with torch.no_grad():
            features = self.nn(img_tensor).last_hidden_state
            features = features.mean(dim=1)
        return F.normalize(features, dim=-1)

    def score_sample(self, f1, f2):
        feat1 = self._extract_features(f1).detach().to('cpu').numpy()
        feat2 = self._extract_features(f2).detach().to('cpu').numpy()
        sim = 1-spatial.distance.cosine(feat1[0], feat2[0])
        return sim
        
    def score_gt(self, folder1, folder2):
        files1 = folder1
        files2 = folder2

        assert len(files1) == len(files2), "Both folders must have the same number of images."

        scores = []
        for f1, f2 in zip(files1, files2):
            feat1 = self._extract_features(f1).detach().to('cpu').numpy()
            feat2 = self._extract_features(f2).detach().to('cpu').numpy()
            sim = 1-spatial.distance.cosine(feat1[0], feat2[0])
            scores.append(sim)

        return np.mean(scores), scores

@torch.no_grad()
class CLIPT:
    def __init__(
        self,
        use_gpu=True,
        size=512,
        device=None,
    ):
        import open_clip
        super().__init__()
        self.size = size
        self.results = []
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_model, _, self.transform = open_clip.create_model_and_transforms(
            "ViT-B/32", 
            device=self.device, 
            pretrained='laion2b_s34b_b79k'
        )
    
    def encode(self, image):
        image_input = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input).detach().cpu().float()
        return image_features
    
    def _load_and_preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.encode(img)
    
    def encode_text(self, text):
        """Encode text prompt to features."""
        text_tokens = open_clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).detach().cpu()
        return text_features
    
    def score_sample(self, img_path1, img_path2, caption):
        """
        Compute CLIP text similarity for a single pair of images.
        
        Args:
            img_path1: Path to first image (ground truth)
            img_path2: Path to second image (generated)
            caption: Text caption/prompt for the images
            
        Returns:
            tuple: (generated_score, ground_truth_score)
                - generated_score: similarity between generated image and text
                - ground_truth_score: similarity between GT image and text
        """
        # Encode text
        text_features = self.encode_text(caption)
        
        # Encode images
        gt_features = self._load_and_preprocess(img_path1)
        gen_features = self._load_and_preprocess(img_path2)
        
        # Compute cosine similarities
        gen_clip_score = 1 - spatial.distance.cosine(
            gen_features.view(gen_features.shape[1]), 
            text_features.view(text_features.shape[1])
        )
        
        gt_clip_score = 1 - spatial.distance.cosine(
            gt_features.view(gt_features.shape[1]), 
            text_features.view(text_features.shape[1])
        )
        
        return gen_clip_score, gt_clip_score
    
    def score_gt(self, folder1, folder2, gt_caption):
        """
        Compute CLIP scores for all image pairs with captions.
        
        Args:
            folder1: List of paths to ground truth images
            folder2: List of paths to generated images
            gt_caption: Dictionary mapping sample names to captions
            
        Returns:
            tuple: (mean_score, list_of_scores)
        """
        scores = []
        
        for f1, f2 in zip(folder1, folder2):
            sample = f1.split('/')[-1]
            
            if sample in gt_caption.keys():
                prompt = gt_caption[sample]
                gen_score, gt_score = self.score_sample(f1, f2, prompt)
                scores.append(gen_score)
        
        return np.mean(scores), scores


def load_image(path, size=512):
    """Load and preprocess image for RAFT."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Handle BGRA (transparent) images
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[alpha == 0] = [255, 255, 255]  # transparent -> white
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def generate_optical_flow(source_path, edited_path, output_path="flow.jpg", size=512, device=None):
    """
    Generate and save an optical flow visualization between source and edited images.
    
    Args:
        source_path:  path to the original/source image
        edited_path:  path to the edited/target image
        output_path:  where to save the flow visualization
        size:         resize both images to this square size
    """
    # ── Load model ──────────────────────────────────────────────────────────
    #print("Loading RAFT model...")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raft_large(pretrained=True, progress=True).to(device).eval()

    # ── Load & preprocess images ─────────────────────────────────────────────
    src_np  = load_image(source_path, size)   # H x W x 3, float32 in [0,1]
    edit_np = load_image(edited_path, size)

    # Normalise to [-1, 1] and convert to B x 3 x H x W tensors
    def to_tensor(img_np):
        t = torch.from_numpy((img_np - 0.5) * 2).permute(2, 0, 1).unsqueeze(0)
        return t.float().to(device).contiguous()

    src_t  = to_tensor(src_np)   # (1, 3, H, W)
    edit_t = to_tensor(edit_np)

    # ── Run RAFT ─────────────────────────────────────────────────────────────
    print("Computing optical flow...")
    with torch.no_grad():
        # raft_large returns a list of flow predictions; [-1] is the finest
        flow = model(src_t, edit_t)[-1]   # (1, 2, H, W)

    #print(f"Flow shape : {flow.shape}")
    #print(f"Flow range : [{flow.min():.3f}, {flow.max():.3f}]")

    # ── Visualise & save ─────────────────────────────────────────────────────
    #flow_img = flow_to_image(flow)[0]          # (3, H, W), uint8
    # pil_img  = ToPILImage()(flow_img)
    # pil_img.save(output_path)
    print(f"Flow image saved to: {output_path}")

    # ── Optional: compute scalar flow magnitude ──────────────────────────────
    flow_np  = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()   # H x W x 2
    magnitude = np.sqrt((flow_np ** 2).sum(axis=-1))            # H x W
    mean_mag  = magnitude.mean()
    #print(f"Mean flow magnitude: {mean_mag:.4f} px")

    return flow_np, mean_mag
