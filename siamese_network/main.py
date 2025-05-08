import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

from dataset import SceneObjectDataset, load_and_split_dataset
from siamese_network import SiameseNetwork
from encoders import ResNet18Encoder, ViTEncoder
from embed_processing_modules import CropFusionMLP, SceneComparisonMLP


CRITERION = nn.BCELoss()
TOTAL_EPOCHS = 2
FREEZE_EPOCHS = 5
REGULARIZATION_DECAY = 1e-5
DROPOUT_RATE = 0.3
VIT_ENCODER_DROPOUT = 0.1
# MODEL = SiameseNetwork(ResNet18Encoder(freeze=True), CropFusionMLP(DROPOUT_RATE), SceneComparisonMLP(DROPOUT_RATE))
MODEL = SiameseNetwork(ViTEncoder(freeze=True, dropout=VIT_ENCODER_DROPOUT), CropFusionMLP(dropout=DROPOUT_RATE, input_dim=1152), SceneComparisonMLP(dropout=DROPOUT_RATE, input_dim=896), encode_obj_with_JPM=True)
WD = "."
MODEL_NAME = "ViT"
CROP_TYPE = "visib_crop"


def evaluate_model(model, dataloader, criterion, device, desc="Eval", is_print=False):
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
        model, criterion, train_loader, val_seen_loader, val_unseen_loader, num_epochs=20, freeze_epochs=5,
        lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=REGULARIZATION_DECAY)

    def freeze_encoder(model, freeze=True):
        for param in model.encoder.parameters():
            param.requires_grad = not freeze

    freeze_encoder(model, freeze=True)

    statistics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
        "val_seen_loss": [],
        "val_seen_acc": [],
        "val_seen_prec": [],
        "val_seen_rec": [],
        "val_seen_f1": [],
        "val_unseen_loss": [],
        "val_unseen_acc": [],
        "val_unseen_prec": [],
        "val_unseen_rec": [],
        "val_unseen_f1": []
    }
    for epoch in range(num_epochs):
        # unfreeze encoder after freeze_epochs to train it in tandem with FusionMLP and ClassifierMLP
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

        val_seen_loss, val_seen_acc, val_seen_prec, val_seen_rec, val_seen_f1 = evaluate_model(model, val_seen_loader, criterion, device)
        val_unseen_loss, val_unseen_acc, val_unseen_prec, val_unseen_rec, val_unseen_f1 = evaluate_model(model, val_unseen_loader, criterion, device)

        total_seen = len(val_seen_loader.dataset)
        total_unseen = len(val_unseen_loader.dataset)
        total = total_seen + total_unseen
        val_loss = (val_seen_loss * total_seen + val_unseen_loss * total_unseen) / total
        val_acc = (val_seen_acc * total_seen + val_unseen_acc * total_unseen) / total
        val_prec = (val_seen_prec * total_seen + val_unseen_prec * total_unseen) / total
        val_rec = (val_seen_rec * total_seen + val_unseen_rec * total_unseen) / total
        val_f1 = (val_seen_f1 * total_seen + val_unseen_f1 * total_unseen) / total


        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}\n"
              f"Validation - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}\n"
              f"    Seen   - Loss: {val_seen_loss:.4f} | Acc: {val_seen_acc:.4f} | Precision: {val_seen_prec:.4f} | Recall: {val_seen_rec:.4f} | F1: {val_seen_f1:.4f}\n"
              f"    Unseen - Loss: {val_unseen_loss:.4f} | Acc: {val_unseen_acc:.4f} | Precision: {val_unseen_prec:.4f} | Recall: {val_unseen_rec:.4f} | F1: {val_unseen_f1:.4f}\n"
              )

        torch.save(model.state_dict(), f"{WD}/output/{MODEL_NAME}{epoch+1}.pth")

        statistics["train_loss"].append(avg_train_loss)
        statistics["val_loss"].append(val_loss)
        statistics["val_acc"].append(val_acc)
        statistics["val_prec"].append(val_prec)
        statistics["val_rec"].append(val_rec)
        statistics["val_f1"].append(val_f1)
        statistics["val_seen_loss"].append(val_seen_loss)
        statistics["val_seen_acc"].append(val_seen_acc)
        statistics["val_seen_prec"].append(val_seen_prec)
        statistics["val_seen_rec"].append(val_seen_rec)
        statistics["val_seen_f1"].append(val_seen_f1)
        statistics["val_unseen_loss"].append(val_unseen_loss)
        statistics["val_unseen_acc"].append(val_unseen_acc)
        statistics["val_unseen_prec"].append(val_unseen_prec)
        statistics["val_unseen_rec"].append(val_unseen_rec)
        statistics["val_unseen_f1"].append(val_unseen_f1)

    return statistics


def graph_statistics(statistics, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    # Create a figure with 5 subplots stacked vertically
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    fig.tight_layout(pad=5.0)

    # 1. Losses
    axes[0].plot(statistics["train_loss"], label="Train Loss")
    axes[0].plot(statistics["val_loss"], label="Val Loss")
    axes[0].plot(statistics["val_seen_loss"], label="Seen Loss")
    axes[0].plot(statistics["val_unseen_loss"], label="Unseen Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Accuracy
    axes[1].plot(statistics["val_acc"], label="Val Acc")
    axes[1].plot(statistics["val_seen_acc"], label="Seen Acc")
    axes[1].plot(statistics["val_unseen_acc"], label="Unseen Acc")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Precision
    axes[2].plot(statistics["val_prec"], label="Val Prec")
    axes[2].plot(statistics["val_seen_prec"], label="Seen Prec")
    axes[2].plot(statistics["val_unseen_prec"], label="Unseen Prec")
    axes[2].set_title("Precision over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Precision")
    axes[2].legend()
    axes[2].grid(True)

    # 4. Recall
    axes[3].plot(statistics["val_rec"], label="Val Rec")
    axes[3].plot(statistics["val_seen_rec"], label="Seen Rec")
    axes[3].plot(statistics["val_unseen_rec"], label="Unseen Rec")
    axes[3].set_title("Recall over Epochs")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Recall")
    axes[3].legend()
    axes[3].grid(True)

    # 5. F1 Score
    axes[4].plot(statistics["val_f1"], label="Val F1")
    axes[4].plot(statistics["val_seen_f1"], label="Seen F1")
    axes[4].plot(statistics["val_unseen_f1"], label="Unseen F1")
    axes[4].set_title("F1 Score over Epochs")
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("F1 Score")
    axes[4].legend()
    axes[4].grid(True)

    # Save the figure
    plot_path = os.path.join(plot_dir, "metrics_over_epochs.png")
    plt.savefig(plot_path)

    # Show the figure in one window
    plt.show()


if __name__ == "__main__":
    results_json_path = f'{WD}/output/results.json'
    object_crop_root = f'{WD}/data/objects/{CROP_TYPE}'
    scene_image_root = f'{WD}/bop_dataset/test'
    graph_root = f'{WD}/output/plots/'

    # prepare dataset
    train, val_seen, val_unseen, test_seen, test_unseen = load_and_split_dataset(results_json_path)
    train_dataset = SceneObjectDataset(train, object_crop_root, scene_image_root)
    val_seen_dataset = SceneObjectDataset(val_seen, object_crop_root, scene_image_root)
    val_unseen_dataset = SceneObjectDataset(val_unseen, object_crop_root, scene_image_root)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_seen_loader = DataLoader(val_seen_dataset, batch_size=32, shuffle=False)
    val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=32, shuffle=False)

    # train
    statistics = train_model(MODEL, CRITERION, train_loader, val_seen_loader, val_unseen_loader, num_epochs=TOTAL_EPOCHS, freeze_epochs=FREEZE_EPOCHS)
    graph_statistics(statistics, graph_root)

    # test
    test_seen_dataset = SceneObjectDataset(val_seen, object_crop_root, scene_image_root)
    test_unseen_dataset = SceneObjectDataset(val_unseen, object_crop_root, scene_image_root)
    test_seen_loader = DataLoader(val_seen_dataset, batch_size=32, shuffle=False)
    test_unseen_loader = DataLoader(val_unseen_dataset, batch_size=32, shuffle=False)

    print("\n======== Final Testing ========")
    test_model_path = input("Enter the path of the model to test: ")
    test_model = torch.load(test_model_path)
    evaluate_model(test_model, test_seen_loader, CRITERION, 'cuda' if torch.cuda.is_available() else 'cpu', desc="  Seen Test Set", is_print=True)
    evaluate_model(test_model, test_unseen_loader, CRITERION, 'cuda' if torch.cuda.is_available() else 'cpu', desc="Unseen Test Set", is_print=True)



