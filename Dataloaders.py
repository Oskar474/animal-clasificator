import os
import re
from pathlib import Path
import pandas as pd
from PIL.Image import Image
from torchvision.io import decode_image
from torch.utils.data import Dataset
import numpy as np

class AnimalDataloader(Dataset):
    def __init__(self, dataframe, data_path, transforms=None):
        self.data_path = Path(data_path)
        self.dataframe = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        base = str(row.iloc[0])
        img_path = self.data_path / f"{base}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        label = int(row.iloc[1])

        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label
