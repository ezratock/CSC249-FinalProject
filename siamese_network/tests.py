import torch
from torch.utils.data import DataLoader, Dataset

from siamese_network import SiameseNetwork
from embed_processing_modules import CropFusionMLP, SceneComparisonMLP
from dataset import SceneObjectDataset, load_and_split_dataset
from encoders import ResNet18Encoder, ViTEncoder

WD="."

def testResNet18Encoder():
    model = ResNet18Encoder(freeze=True)
    dummy_image = torch.randn(1, 3, 224, 224)
    output = model(dummy_image)
    assert output.shape == torch.Size([1, 512])
    print("testResNet18Encoder() passed")

def testViTEncoder():
    model = ViTEncoder(freeze=True)
    dummy_image = torch.randn(1, 3, 224, 224)
    output = model(dummy_image, use_jpm=False)
    assert output.shape == torch.Size([1, 384])
    output = model(dummy_image, use_jpm=True)
    assert output.shape == torch.Size([1, 384*3])
    print("testViTEncoder() passed")

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


def testFullModelHelper(model, B):
    model.eval()
    C, height, width = 3, 224, 224
    crop_1 = torch.rand(B, C, height, width)
    crop_2 = torch.rand(B, C, height, width)
    crop_3 = torch.rand(B, C, height, width)
    scene = torch.rand(B, C, height, width)
    return model(crop_1, crop_2, crop_3, scene)


def testResNet18FullModel():
    model = SiameseNetwork(ResNet18Encoder(freeze=True), CropFusionMLP(), SceneComparisonMLP())
    B=5
    output = testFullModelHelper(model, B)
    assert output.shape == torch.Size([B, 1])
    print("testResNet18FullModel() passed")


def testViTFullModel():
    model = SiameseNetwork(ViTEncoder(freeze=True), CropFusionMLP(input_dim=1152), SceneComparisonMLP(input_dim=896), encode_obj_with_JPM=True)
    B=5
    output = testFullModelHelper(model, B)
    assert output.shape == torch.Size([B, 1])
    print("testViTFullModel() passed")


def testDatasetPrep():
    train, val_seen, val_unseen, test_seen, test_unseen = load_and_split_dataset(f"{WD}/output/results.json")
    for name, value in {"train": train, "val_seen": val_seen, "val_unseen": val_unseen, "test_seen": test_seen, "test_unseen": test_unseen}.items():
        positive = [t for t in value if t[3] == 1]
        negative = [t for t in value if t[3] == 0]
        assert len(positive) == len(negative)
        print(f"{name} samples: {len(value)} | positive: {len(positive)} | negative: {len(negative)}")

    train_dataset = SceneObjectDataset(train, f"{WD}/data/objects/full_crop", f"{WD}/bop_dataset/test")
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


if __name__ == "__main__":
    testResNet18Encoder()
    testViTEncoder()
    testCropFusion()
    testSceneComparison()
    testDatasetPrep()
    testResNet18FullModel()
    testViTFullModel()