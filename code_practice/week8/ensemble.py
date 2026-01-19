from ddfs_classify import CustomDataset, normalized_channel_means, normalized_channel_stds, set_transform_compose, set_seed, test_df
import os, random, cv2, shutil, glob, gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from torchvision import models
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image


def load_model(model_id, device):
    model_path = f"./models/best_model_{model_id}.pth"
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(model.classifier[2].in_features, 2)
    )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def set_dataloader(test_df, batch_size):
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalized_channel_means, std=normalized_channel_stds)
    ])

    test_dataset = CustomDataset(dataframe=test_df, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


@torch.no_grad()
def ensemble_eval(model_ids, test_loader, weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(model_id=model_id, device=device) for model_id in model_ids]

    if weights is None:
        weights = np.ones(len(models), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    all_y = []
    all_logits = []

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        logits_sum = None

        for w, m in  zip(weights, models):
            logits = m(inputs)
            logits_sum = logits * w if logits_sum is None else logits_sum + logits * w

        all_logits.append(logits_sum.cpu())
        all_y.append(labels.cpu())


    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    probs_fake = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = torch.argmax(logits, dim=1).numpy()

    acc = float((y_true == y_pred).mean())
    auc = float("nan")
    if len(np.unique(y_true)) == 2:  # ✅ 한 클래스면 roc_auc_score 에러 방지
        auc = float(roc_auc_score(y_true, probs_fake))    
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Ensemble Test Confusion Matrix")
    fig.tight_layout()

    f1 = f1_score(y_true, y_pred, average="weighted")

    return acc, auc, f1, fig

def main():
    batch_size = 32
    model_ids = ["3zmbk7kj", "rebnyi53", "wfubby3q", "w5qayyi3", "ooejv2kk"]
    weight = None # Equal weights
    test_loader = set_dataloader(test_df=test_df, batch_size=batch_size)
    acc, auc, f1, fig = ensemble_eval(model_ids=model_ids, test_loader=test_loader, weights=weight)

    wandb.init(project="ConvNeXt-only", entity="DDFS", name="Ensemble_Eval")
    wandb.log({
        "ensemble_k": 5,
        "ensemble_weighted": int(weight is not None),
        "test_acc": acc,
        "test_auc": auc,
        "test_f1": f1,
        "confusion_matrix": wandb.Image(fig)
    })
    wandb.finish()
    plt.close(fig)

    print(f"Ensemble Test Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()