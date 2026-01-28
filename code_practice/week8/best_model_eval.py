import os
import glob
import ast
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt


# 통계 값은 ddfs_classify.py와 동일하게 사용
normalized_channel_means = [0.5159901929579159, 0.43138779010237205, 0.38887346908831566]
normalized_channel_stds = [0.2922462811984076, 0.2667907590120976, 0.26435657587983385]

MODEL_DIR = "./models"
CONFIG_DIR = "./configs"


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, return_path: bool = False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = 0 if self.dataframe.iloc[idx, 1] == "real" else 1
        if self.return_path:
            return img, label, img_path
        return img, label


def set_transform(normalized_means, normalized_stds):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalized_means, std=normalized_stds)
    ])


def load_config(best_id: str):
    cfg_path = os.path.join(CONFIG_DIR, f"best_config_{best_id}.txt")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return ast.literal_eval(f.read())


def set_model(config):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=float(config.get("dropout", 0.0))),
        nn.Linear(model.classifier[2].in_features, 2)
    )
    return model


def build_test_df():
    test_image_paths = glob.glob("./dataset/test/*/*.jpg")
    if not test_image_paths:
        raise FileNotFoundError("No test images found under ./dataset/test/*/*.jpg. Build the dataset first (refer to ddfs_classify.py).")
    test_df = pd.DataFrame({"path": test_image_paths})
    test_df["label"] = test_df["path"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    return test_df


@torch.no_grad()
def eval_best(model, test_loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    misclassified = []

    for batch in test_loader:
        if len(batch) == 2:
            inputs, labels = batch
            paths = None
        else:
            inputs, labels, paths = batch

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

        if paths is not None:
            wrong_mask = (preds != labels).cpu().numpy()
            for p, t, pr, wrong in zip(paths, labels.cpu().numpy(), preds.cpu().numpy(), wrong_mask):
                if wrong:
                    misclassified.append((p, int(t), int(pr)))

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = (y_pred == y_true).mean()
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, average="binary")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Best Model Test Confusion Matrix")
        ax.set_xticks([0, 1], labels=["Real", "Fake"])
        ax.set_yticks([0, 1], labels=["Real", "Fake"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
    except Exception:
        fig = None

    return float(acc), float(auc), float(f1), fig, misclassified


def run(best_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(best_id)
    model_path = os.path.join(MODEL_DIR, f"best_model_{best_id}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = set_model(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_df = build_test_df()
    transform = set_transform(normalized_channel_means, normalized_channel_stds)
    test_dataset = CustomDataset(test_df, transform=transform, return_path=True)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, num_workers=4)

    acc, auc, f1, fig, mis = eval_best(model, test_loader, device)
    print(f"[BEST TEST] Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
    if mis:
        print("Misclassified samples (path, true_label, pred_label):")
        for p, t, pr in mis:
            print(p, t, pr)
    if fig:
        plt.close(fig)


if __name__ == "__main__":
    best_id_env = os.environ.get("BEST_ID")
    if not best_id_env:
        raise ValueError("Set BEST_ID environment variable (e.g., BEST_ID=your_run_id) to run best model evaluation.")
    run(best_id_env)
