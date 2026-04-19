import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import  torch
from Dataloaders import AnimalDataloader
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.CNN_2 import CNN
from models.CNN_3 import PaperCNN_Upgraded

train_transforms = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=0,
        fill=(0, 0, 0)
    ),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=0,
        fill=(0, 0, 0)
    ),
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

    print(df.head())

    return df

import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels, class_names=None, n=6):
    images = images[:n]
    labels = labels[:n]

    fig, axes = plt.subplots(1, n, figsize=(15, 5))

    for i in range(n):
        img = images[i].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW → HWC

        # unnormalize (IMPORTANT!)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        title = labels[i].item()
        if class_names:
            title = class_names[title]

        axes[i].set_title(title)
        axes[i].axis("off")

    plt.show()

def split_data(data_df):
    train_df, val_df = train_test_split(data_df, train_size=0.8)
    # val_df, test_df = train_test_split(val_df, train_size=0.666)
    test_df = None
    return  train_df, val_df, test_df

def create_datasets(train_df, val_df, test_df):
    val_ds = AnimalDataloader(val_df, r'data/images', val_transforms)
    train_ds = AnimalDataloader(train_df, r'data/images', train_transforms)
    test_ds = AnimalDataloader(test_df, r'data/images', val_transforms)
    return val_ds, train_ds, test_ds

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

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

def get_model():

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,32,3, padding=1),
                nn.MaxPool2d(2),
            )
    return CNN

def plot_training_curves(history, image_name):
    sns.set_theme('paper')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=history['train_acc'], label='Train Acc', ax=ax[0])
    sns.lineplot(data=history['val_acc'], label='Val Acc', ax=ax[0])
    ax[0].set_title('Accuracy')

    sns.lineplot(data=history['train_loss'], label='Train Loss', ax=ax[1])
    sns.lineplot(data=history['val_loss'], label='Val Loss', ax=ax[1])
    ax[1].set_title('Loss')


    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / image_name)
    plt.show()

if __name__ == "__main__":
    data_df = load_data()
    train_df, val_df, test_df = split_data(data_df)
    val_ds, train_ds, test_ds = create_datasets(train_df, val_df, test_df)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)

    # images, labels = next(iter(train_loader))
    # images, labels = next(iter(train_loader))
    # show_images(images, labels)

    model = PaperCNN_Upgraded()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(30):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)


        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

    plot_training_curves(history, "CNN_1.jpg")
