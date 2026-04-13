import pandas as pd
from sklearn.model_selection import train_test_split
from Dataloaders import AnimalDataloader
import torch.nn as nn

def load_data():
    data_df = pd.read_csv(r'data/annotations/list.txt')
    return data_df

def split_data(data_df):
    train_df, val_df = train_test_split(data_df, train_size=0.7)
    val_df, test_df = train_test_split(val_df, train_size=0.666)
    return  train_df, val_df, test_df

def create_datasets(train_df, val_df, test_df):
    val_ds = AnimalDataloader(val_df, r'data/images')
    train_ds = AnimalDataloader(train_df, r'data/images')
    test_ds = AnimalDataloader(test_df, r'data/images')
    return val_ds, train_ds, test_ds

def train_one_epoch():
    pass

def validate():
    pass

def get_model():

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,32,3, padding=1),

                nn.MaxPool2d(2),
            )

    pass

if __name__ == "__main__":
    data_df = load_data()
    train_df, val_df, test_df = split_data(data_df)
    val_ds, train_ds, test_ds = create_datasets(train_df, val_df, test_df)

# def train_validate(model, train_fd, val_df, epochs):
#     pass