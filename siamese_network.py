import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import random
import os
import glob
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


class ResNet18Encoder(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()

        # pretrained ImageNet weights
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # remove the final ImageNet classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 512, 1, 1)
        x = self.flatten(x)            # (B, 512)
        return x


class CropFusionMLP(nn.Module):
    def __init__(self, input_dim=512, fused_dim=512):
        super().__init__()

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 3, 1024),  # 1536 to 1024
            nn.ReLU(),
            nn.Linear(1024, fused_dim),     # 1024 to 512
            nn.ReLU()
        )

    def forward(self, crop1, crop2, crop3):
        x = torch.cat([crop1, crop2, crop3], dim=1)  # (B, 1536)
        fused = self.fusion_mlp(x)                   # (B, 512)
        return fused


class SceneComparisonMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, obj_fusion, scene_embedding):
        x = torch.cat([obj_fusion, scene_embedding], dim=1)  # concatenate: (B, 1024)
        return self.classifier(x)  # output: (B, 1)


class SiameseNetwork(nn.Module):
    def __init__(self, freeze_encoder=True):
        super(SiameseNetwork, self).__init__()
        self.encoder = ResNet18Encoder(freeze=freeze_encoder)
        self.crop_fusion_mlp = CropFusionMLP()
        self.scene_comparison_mlp = SceneComparisonMLP()

    def forward(self, crop1, crop2, crop3, scene):
        """
        crop1, crop2, crop3: (B, 3, H, W)
        scene: (B, 3, H, W)
        Returns: (B, 1) — binary output between 0 and 1
        """

        # embed all crops and scene with shared ResNet18 encoder
        embed_1 = self.encoder(crop1)   # (B, 512)
        embed_2 = self.encoder(crop2)   # (B, 512)
        embed_3 = self.encoder(crop3)   # (B, 512)
        scene_embed = self.encoder(scene)  # (B, 512)

        # fuse all crops with MLP
        fused_crop = self.crop_fusion_mlp(embed_1, embed_2, embed_3)  # (B, 512)

        # predict if the object is in the scene based on embeddings with MLP
        output = self.scene_comparison_mlp(fused_crop, scene_embed)  # (B, 1)

        return output

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


# def load_and_split_dataset(
#         results_path: str = "output/results.json",
#         heldout_scenes: List[str] = ["000052", "000054"],  # scenes reserved fully for validation
#         train_ratio: float = 0.9,
#         seed: int = 42
# ) -> Tuple[List[Tuple[str, str, str, int]], List[Tuple[str, str, str, int]]]:
#     """
#     Loads results.json and returns two lists of samples:
#     (scene_num, frame_num, object_name, true_label)
#     """
#
#     with open(results_path, "r") as f:
#         data = json.load(f)
#
#     all_scenes = sorted(data.keys())
#     main_scenes = [s for s in all_scenes if s not in heldout_scenes]
#
#     random.seed(seed)
#     train_samples = []
#     val_samples = []
#
#     for scene in main_scenes:
#         frame_dict = data[scene]
#         frames = sorted(frame_dict.keys())
#         random.shuffle(frames)
#
#         num_train = int(len(frames) * train_ratio)
#         train_frames = frames[:num_train]
#         val_frames = frames[num_train:]
#
#         # add 90% of main scenes to training
#         for frame in train_frames:
#             for obj_data in frame_dict[frame]:
#                 train_samples.append((scene, frame, obj_data["object"], obj_data["true_label"]))
#
#         # add remaining 10% of main scenes to validation
#         for frame in val_frames:
#             for obj_data in frame_dict[frame]:
#                 val_samples.append((scene, frame, obj_data["object"], obj_data["true_label"]))
#
#     # add 100% of held out scenes to validation
#     for scene in heldout_scenes:
#         frame_dict = data[scene]
#         for frame in frame_dict:
#             for obj_data in frame_dict[frame]:
#                 val_samples.append((scene, frame, obj_data["object"], obj_data["true_label"]))
#
#     return train_samples, val_samples

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

    for frame, objs in data[scene_val_unseen].items():
        for obj in objs:
            val_unseen.append((scene_val_unseen, frame, obj["object"], obj["true_label"]))
    for frame, objs in data[scene_test_unseen].items():
        for obj in objs:
            test_unseen.append((scene_val_unseen, frame, obj["object"], obj["true_label"]))

    return train, val_seen, val_unseen, test_seen, test_unseen

def evaluate_model(model, dataloader, criterion, device, desc="Eval", is_print = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(dataloader, desc=desc, leave=True)
        for crop1, crop2, crop3, scene, label in loop:
            crop1 = crop1.to(device)
            crop2 = crop2.to(device)
            crop3 = crop3.to(device)
            scene = scene.to(device)
            label = label.float().unsqueeze(1).to(device)

            output = model(crop1, crop2, crop3, scene)
            loss = criterion(output, label)

            total_loss += loss.item() * crop1.size(0)

            preds = (output > 0.5).float()
            total_correct += (preds == label).sum().item()
            total_samples += label.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    if is_print:
        print(f"{desc} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
          f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return avg_loss, accuracy, precision, recall, f1

def train_model(
        model, train_loader, val_loader, num_epochs=20, freeze_epochs=5,
        lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def freeze_encoder(model, freeze=True):
        for param in model.encoder.parameters():
            param.requires_grad = not freeze

    freeze_encoder(model, freeze=True)

    for epoch in range(num_epochs):
        # unfreeze encoder after freeze_epochs to train it in tandum with FusionMLP and ClassifierMLP
        if epoch == freeze_epochs:
            freeze_encoder(model, freeze=False)
            print("Unfroze encoder...")

        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)
        for crop1, crop2, crop3, scene, label in train_loop:
            crop1 = crop1.to(device)
            crop2 = crop2.to(device)
            crop3 = crop3.to(device)
            scene = scene.to(device)
            label = label.float().unsqueeze(1).to(device)  # (B, 1)

            optimizer.zero_grad()
            output = model(crop1, crop2, crop3, scene)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * crop1.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}) "
              )

    return model


def testEncoder():
    model = ResNet18Encoder(freeze=True)
    dummy_image = torch.randn(1, 3, 224, 224)
    output = model(dummy_image)
    assert output.shape == torch.Size([1, 512])
    print("testEncoder() passed")

def testCropFusion():
    model = CropFusionMLP()
    crop_embed_1 = torch.rand(1, 512)
    crop_embed_2 = torch.rand(1, 512)
    crop_embed_3 = torch.rand(1, 512)
    output = model(crop_embed_1, crop_embed_2, crop_embed_3)
    assert output.shape == torch.Size([1, 512])
    print("testCropFusion() passed")

def testSceneComparison():
    model = SceneComparisonMLP()
    fused_crop = torch.rand(5, 512)
    scene_embed = torch.rand(5, 512)
    output = model(fused_crop, scene_embed)
    assert output.shape == torch.Size([5, 1])
    assert torch.all((output >= 0) & (output <= 1))
    print("testSceneComparison() passed")

def testFullModel():
    model = SiameseNetwork();
    model.eval()

    B, C, height, width = 5, 3, 224, 224
    crop_1 = torch.rand(B, C, height, width)
    crop_2 = torch.rand(B, C, height, width)
    crop_3 = torch.rand(B, C, height, width)
    scene = torch.rand(B, C, height, width)

    output = model(crop_1, crop_2, crop_3, scene)
    assert output.shape == torch.Size([B, 1])
    print("testFullMode() passed")

def testDatasetPrep():
    train, val_seen, val_unseen, test_seen, test_unseen = load_and_split_dataset()
    for name, value in {"train": train, "val_seen": val_seen, "val_unseen": val_unseen, "test_seen": test_seen, "test_unseen": test_unseen}.items():
        positive = [t for t in value if t[3] == 1]
        negative = [t for t in value if t[3] == 0]
        assert len(positive) == len(negative)
        print(f"{name} samples: {len(value)} | positive: {len(positive)} | negative: {len(negative)}")

    train_dataset = SceneObjectDataset(train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # single batch
    for crop1, crop2, crop3, scene, label in train_loader:
        assert crop1.shape == torch.Size([2, 3, 224, 224])
        assert crop2.shape == torch.Size([2, 3, 224, 224])
        assert crop3.shape == torch.Size([2, 3, 224, 224])
        assert scene.shape == torch.Size([2, 3, 224, 224])
        # print(label.shape)
        assert label.shape == torch.Size([2])
        break

    print("testDatasetPrep() passed")


def main():
    results_json_path = 'output/results.json'
    object_crop_root = 'data/objects/full_crop'
    scene_image_root = 'bop_dataset/test'

    # prepare dataset
    train_data, val_data = load_and_split_dataset(results_json_path)
    train_dataset = SceneObjectDataset(train_data, object_crop_root, scene_image_root)
    val_dataset = SceneObjectDataset(val_data, object_crop_root, scene_image_root)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # train
    model = SiameseNetwork()
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)
    torch.save(trained_model.state_dict(), 'output/siamese_network.pth')

if __name__ == "__main__":
    # testEncoder()
    # testCropFusion()
    # testSceneComparison()
    testDatasetPrep()
    # testFullModel()

    # main()
