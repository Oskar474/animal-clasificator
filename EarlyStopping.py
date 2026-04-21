import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("New best model")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True