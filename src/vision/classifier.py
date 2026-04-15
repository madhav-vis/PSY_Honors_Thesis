"""CLIP zero-shot and fine-tuned classification of gaze-cropped images."""

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm

from .config import CATEGORIES, CLIP_MODEL


class GazeClassifier:
    """Wraps CLIP for zero-shot or fine-tuned gaze crop classification.

    Text features for all categories are pre-encoded once at init.
    Optionally loads a trained linear head (.pt) for fine-tuned
    classification that replaces text-similarity scoring.
    """

    def __init__(self, model_name=CLIP_MODEL, categories=None,
                 head_path=None):
        if categories is None:
            categories = CATEGORIES

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"    CLIP device: {self.device}")

        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP model '{model_name}'. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self.labels = list(categories.keys())
        self.prompts = list(categories.values())

        text_tokens = clip.tokenize(self.prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

        self.head = None
        self.head_stats = None
        if head_path is not None:
            self.load_head(head_path)

    def load_head(self, head_path):
        """Load a trained linear classification head from a .pt file."""
        from .train_head import load_head
        self.head, self.head_stats = load_head(head_path, n_classes=len(self.labels))
        self.head = self.head.to(self.device)
        self.head.eval()
        acc = self.head_stats.get("train_acc", 0)
        print(f"    Loaded fine-tuned head from {head_path} "
              f"(train acc: {acc:.3f})")

    @property
    def has_head(self):
        return self.head is not None

    def classify_crop(self, crop_rgb_np):
        """Classify a single RGB crop.

        Uses fine-tuned head if loaded, otherwise zero-shot.
        Returns (label, confidence, all_scores_dict).
        """
        pil_img = Image.fromarray(crop_rgb_np)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        if self.head is not None:
            with torch.no_grad():
                logits = self.head(image_features.float())
                probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
        else:
            similarity = (image_features @ self.text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1).cpu().numpy()

        best_idx = int(np.argmax(probs))
        all_scores = {lbl: float(p) for lbl, p in zip(self.labels, probs)}
        return self.labels[best_idx], float(probs[best_idx]), all_scores

    def classify_batch(self, crops, batch_size=32):
        """Classify a list of RGB crop arrays.

        Uses fine-tuned head if loaded, otherwise zero-shot.
        Returns list of dicts with keys: label, confidence, all_scores.
        """
        mode = "fine-tuned head" if self.head is not None else "zero-shot"
        results = []
        for start in tqdm(range(0, len(crops), batch_size),
                          desc=f"Classifying ({mode})"):
            batch = crops[start : start + batch_size]
            images = torch.stack(
                [self.preprocess(Image.fromarray(c)) for c in batch]
            ).to(self.device)

            with torch.no_grad():
                img_feats = self.model.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            if self.head is not None:
                with torch.no_grad():
                    logits = self.head(img_feats.float())
                    probs = logits.softmax(dim=-1).cpu().numpy()
            else:
                sims = (img_feats @ self.text_features.T)
                probs = sims.softmax(dim=-1).cpu().numpy()

            for p in probs:
                best_idx = int(np.argmax(p))
                all_scores = {lbl: float(v) for lbl, v in zip(self.labels, p)}
                results.append(
                    {
                        "label": self.labels[best_idx],
                        "confidence": float(p[best_idx]),
                        "all_scores": all_scores,
                    }
                )

        return results

    # CLIP embeddings: 512-dim L2-normalized image representations.
    # Similar visual content → similar embeddings (high cosine similarity).
    # Used instead of zero-shot classification for egocentric footage
    # where named categories perform at chance level.

    def extract_embedding(self, crop_rgb_np):
        """Extract a single L2-normalized CLIP image embedding.

        Returns 1D numpy array of shape (512,) float32.
        """
        pil_img = Image.fromarray(crop_rgb_np)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(image_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_embeddings_batch(self, crops, batch_size=32):
        """Extract L2-normalized CLIP embeddings for a list of crops.

        Returns numpy array of shape (n_crops, 512) float32.
        """
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        all_embs = []
        for start in tqdm(range(0, len(crops), batch_size),
                          desc="Extracting embeddings"):
            batch = crops[start : start + batch_size]
            images = torch.stack(
                [self.preprocess(Image.fromarray(c)) for c in batch]
            ).to(self.device)

            with torch.no_grad():
                feats = self.model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            all_embs.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embs, axis=0)
