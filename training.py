import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from Dataloaders import AnimalDataloader
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from EarlyStopping import EarlyStopping
from models.ResidualCNN import ResidualCNN
from models.CNN_1 import CNN1
from models.CNN_2 import CNN2
from models.CNN_3 import CNN3
from models.CNN_4 import CNN4
from models.CNN_5 import CNN5

train_transforms = A.Compose([
    A.LongestMaxSize(224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, fill=(0, 0, 0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=10,
        p=0.3
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, fill=(0, 0, 0)),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def load_data():
    df = pd.read_csv(
        r"data/annotations/list.txt",
        sep=" ",
        header=None,
        comment="#"
    )

    df = df[[0, 1]]
    df.columns = ["image", "label"]
    df["label"] = df["label"] - 1

    class_names = (
        df.groupby("label")["image"]
        .first()
        .sort_index()
        .apply(lambda x: x.split("_")[0])
        .tolist()
    )
    print(class_names)
    return df


def split_data(data_df):
    train_df, val_df = train_test_split(data_df, train_size=0.8, stratify=data_df["label"])
    return train_df, val_df


def create_datasets(train_df, val_df):
    train_ds = AnimalDataloader(train_df, r'data/images', train_transforms)
    val_ds = AnimalDataloader(val_df, r'data/images', val_transforms)
    return train_ds, val_ds


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)



def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_training_curves(history, image_name):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(len(history["train_acc"]))

    sns.lineplot(x=epochs, y=history['train_acc'], label='Train Acc', ax=ax[0])
    sns.lineplot(x=epochs, y=history['val_acc'], label='Val Acc', ax=ax[0])
    ax[0].set_title('Accuracy')

    sns.lineplot(x=epochs, y=history['train_loss'], label='Train Loss', ax=ax[1])
    sns.lineplot(x=epochs, y=history['val_loss'], label='Val Loss', ax=ax[1])
    ax[1].set_title('Loss')

    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / image_name)
    plt.show()

def compute_metrics(labels, preds):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
    }
    return metrics

if __name__ == "__main__":
    data_df = load_data()
    train_df, val_df = split_data(data_df)

    train_ds, val_ds = create_datasets(train_df, val_df)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = CNN5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(50):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            break

    plot_training_curves(history, "EXP_9.jpg")

    model.load_state_dict(torch.load("best_model.pth"))
    labels, preds = get_predictions(model, val_loader, device)

    metrics = compute_metrics(labels, preds)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
