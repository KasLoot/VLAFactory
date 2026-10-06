import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class VAEDataset(Dataset):
    """Load RGB images from one split directory and normalize to [-1, 1].

    The manifest is a list of image records. Prefer label.json when present;
    otherwise use the existing post_processed_images-labels.json filename.
    Metadata provides filenames and dimensions, not supervised targets.
    """

    def __init__(self, dataset_path=None):
        if dataset_path is None or (
            isinstance(dataset_path, str) and not dataset_path.strip()
        ):
            raise ValueError("dataset_path must be provided and point to a split directory")
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset split directory does not exist: {self.dataset_path}")
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset split path is not a directory: {self.dataset_path}")
        for filename in ("label.json", "post_processed_images-labels.json"):
            manifest_path = self.dataset_path / filename
            if manifest_path.is_file():
                break
        else:
            raise FileNotFoundError(
                f"Expected label.json or post_processed_images-labels.json "
                f"in {self.dataset_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            self.dataset_json = json.load(file)
        if not isinstance(self.dataset_json, list):
            raise ValueError(f"Expected a list of image records in {manifest_path}")
        self.dataset_size = len(self.dataset_json)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        record = self.dataset_json[idx]
        with Image.open(self.dataset_path / record["image"]) as image:
            with image.convert("RGB") as rgb:
                pixels = pil_to_tensor(rgb)
        return pixels.float().div_(127.5).sub_(1.0)
