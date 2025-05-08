import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import json
import random
import os
import glob
from typing import List, Tuple
from PIL import Image


class SceneObjectDataset(Dataset):
    def __init__(
            self,
            samples: List[Tuple[str, str, str, int]],
            crop_dir: str = "data/objects/full_crop",
            scene_dir: str = "bop_dataset/test",
            image_size: int = 224
    ):
        self.samples = samples
        self.crop_dir = crop_dir
        self.scene_dir = scene_dir

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def load_crop_images(self, object_name: str) -> List[torch.Tensor]:
        crop_path = os.path.join(self.crop_dir, object_name)
        crop_files = sorted(glob.glob(os.path.join(crop_path, "*.png")))

        crops = []
        for file in crop_files[:3]:
            image = Image.open(file).convert("RGB")
            crops.append(self.transform(image))
        # if there are less than tree crop images (unexpected), repeat the first crop or use black images
        while len(crops) < 3:
            if crops:
                crops.append(crops[0].clone())
            else:
                crops.append(torch.zeros(3, 224, 224))

        return crops


    def __getitem__(self, idx: int):
        scene, frame, object_name, label = self.samples[idx]

        # load scene image
        scene_img_path = os.path.join(self.scene_dir, scene, "rgb", f"{frame}.png")
        scene_image = Image.open(scene_img_path).convert("RGB")
        scene_tensor = self.transform(scene_image)

        # load crop images
        crop_tensors = self.load_crop_images(object_name)

        # return as a tuple: 3 crops, scene image, label
        return crop_tensors[0], crop_tensors[1], crop_tensors[2], scene_tensor, torch.tensor(label, dtype=torch.float32)

def load_and_split_dataset(
        results_path: str = "output/results.json",
        seed: int = 42
) -> Tuple[List[Tuple[str, str, str, int]],
           List[Tuple[str, str, str, int]],
           List[Tuple[str, str, str, int]],
           List[Tuple[str, str, str, int]],
           List[Tuple[str, str, str, int]]]:
    with open(results_path, "r") as f:
        data = json.load(f)

    all_scenes = sorted(data.keys())
    scene_val_unseen = "000056"
    scene_test_unseen = "000054"
    main_scenes = [s for s in all_scenes if s not in [scene_val_unseen, scene_test_unseen]]

    random.seed(seed)

    train = []
    val_seen = []
    val_unseen = []
    test_seen = []
    test_unseen = []

    # add 80% of main_scenes to training, 10% to validation, and 10% to testing
    for scene in main_scenes:
        positive_samples = []
        negative_samples = []
        for frame, objs in data[scene].items():
            for obj in objs:
                sample = (scene, frame, obj["object"], obj["true_label"])
                if obj["true_label"] == 1:
                    positive_samples.append(sample)
                else:
                    negative_samples.append(sample)

        assert len(positive_samples) == len(negative_samples), f"unequal number of positive and negatives samples in scene {scene}"
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)
        n = len(positive_samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train += positive_samples[:n_train]
        train += negative_samples[:n_train]
        val_seen += positive_samples[n_train:n_train+n_val]
        val_seen += negative_samples[n_train:n_train+n_val]
        test_seen += positive_samples[n_train+n_val:]
        test_seen += negative_samples[n_train+n_val:]

    random.shuffle(train)
    random.shuffle(val_seen)
    random.shuffle(test_seen)

    # add 100% of scene 000056 to validation set and 100% of scene 000054 to test set
    for frame, objs in data[scene_val_unseen].items():
        for obj in objs:
            val_unseen.append((scene_val_unseen, frame, obj["object"], obj["true_label"]))
    for frame, objs in data[scene_test_unseen].items():
        for obj in objs:
            test_unseen.append((scene_val_unseen, frame, obj["object"], obj["true_label"]))

    return train, val_seen, val_unseen, test_seen, test_unseen
