import os
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms


class TestDataset:
    def __init__(self, image_root, gt_root=None, testsize=352):
        self.testsize = testsize
        self.image_root = Path(image_root)
        self.gt_root = Path(gt_root) if gt_root else None
        self.images = sorted(
            [p for p in self.image_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if self.gt_root and self.gt_root.exists():
            self.gts = sorted([p for p in self.gt_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        else:
            self.gts = [None] * len(self.images)
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image_path = self.images[self.index]
        image = self.rgb_loader(image_path)
        image_tensor = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index]) if self.gts[self.index] else image.convert("L")
        name = image_path.with_suffix(".png").name
        self.index = (self.index + 1) % self.size
        return image_tensor, gt, name

    @staticmethod
    def rgb_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    @staticmethod
    def binary_loader(path):
        with open(path, "rb") as f:
            return Image.open(f).convert("L")
