"""ResNet-50 gaze crop classifier."""

import numpy as np
import torch


class ResNetGazeClassifier:
    """Fine-tuned ResNet-50 for gaze crop classification.

    Wraps a saved ResNet-50 checkpoint with a classify_batch() interface
    for use by the vision pipeline and annotator.
    """

    def __init__(self, path, label_names, batch_size=32):
        from .resnet_head import load_resnet
        try:
            from torchvision import transforms
        except ImportError as e:
            raise ImportError(
                "torchvision is required for ResNetGazeClassifier. "
                "Install with: pip install torchvision"
            ) from e

        self.model, self.stats = load_resnet(path)
        self.label_names = label_names
        self.batch_size = batch_size

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print(f"    ResNet classifier loaded from {path} (device={self.device})")

    def classify_batch(self, crops, batch_size=None):
        """Classify a list of RGB crop arrays using ResNet-50.

        Returns list of dicts with keys: label, confidence, all_scores.
        Crops must be 224x224 uint8 numpy arrays (RGB).
        """
        from PIL import Image
        bs = batch_size or self.batch_size
        results = []
        for start in range(0, len(crops), bs):
            batch = crops[start : start + bs]
            imgs = torch.stack([
                self.transform(Image.fromarray(c))
                for c in batch
            ]).to(self.device)
            with torch.no_grad():
                probs = self.model(imgs).softmax(dim=-1).cpu().numpy()
            for p in probs:
                best = int(np.argmax(p))
                results.append({
                    "label": self.label_names[best],
                    "confidence": float(p[best]),
                    "all_scores": {n: float(v)
                                   for n, v in zip(self.label_names, p)},
                })
        return results
