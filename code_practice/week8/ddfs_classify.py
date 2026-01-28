
import os, random, cv2, shutil, glob, gc, ast
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


real = "./real_and_fake_face/training_real/"
fake = "./real_and_fake_face/training_fake/"

real_path = os.listdir(real)
fake_path = os.listdir(fake)


# 3) 데이터셋 분리
# - 현재 data가 training과 test로 분리되어 있지 않음. 따라서 training set과 test set으로 data를 분리하여 라벨링할 것

shutil.rmtree("./dataset", ignore_errors=True)
SRC_ROOT = "./real_and_fake_face"
DST_ROOT = "./dataset"

SPLIT_RATIO = 0.2
SEED = 42

random.seed(SEED)

exts = (".jpg")

pairs = [
    # source, train_dest, test_dest
    ("training_real", "train/real", "test/real"),
    ("training_fake", "train/fake", "test/fake"),
]

# --------------------
# 디렉토리 생성
# --------------------
for _, train_dst, test_dst in pairs:
    os.makedirs(os.path.join(DST_ROOT, train_dst), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, test_dst), exist_ok=True)

# --------------------
# split + 이동
# --------------------
for src_name, train_dst, test_dst in pairs:
    src_dir = os.path.join(SRC_ROOT, src_name)
    train_dir = os.path.join(DST_ROOT, train_dst)
    test_dir  = os.path.join(DST_ROOT, test_dst)
    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(exts)
    ]

    total = len(files)
    num_test = int(total * SPLIT_RATIO)

    random.shuffle(files)
    test_files = files[:num_test]
    train_files = files[num_test:]

    print(f"\n[{src_name}]")
    print(f"총 이미지 수: {total}")
    print(f"training: {len(train_files)}")
    print(f"test: {len(test_files)}")

    # training 이동
    for f in train_files:
        shutil.move(
            os.path.join(src_dir, f),
            os.path.join(train_dir, f)
        )

    # test 이동
    for f in test_files:
        shutil.move(
            os.path.join(src_dir, f),
            os.path.join(test_dir, f)
        )


# 4) dataFrame 생성
# - 수치 데이터(2차원)가 아닌, 이미지 데이터(3차원)이기 때문에, 경로명을 column으로 가진 dataframe을 train과 test별로 제작할 것

train_valid_image_paths = glob.glob("./dataset/train/*/*.jpg")
train_valid_df = pd.DataFrame({"path": train_valid_image_paths})

# label = 상위 폴더명 (real/fake)
train_valid_df["label"] = train_valid_df["path"].apply(lambda x: os.path.basename(os.path.dirname(x)))

print(train_valid_df.head())
print(train_valid_df["label"].value_counts())
print("train rows:", len(train_valid_df))

test_image_paths = glob.glob("./dataset/test/*/*.jpg")
test_df = pd.DataFrame({"path": test_image_paths})

# label = 상위 폴더명 (real/fake)
test_df["label"] = test_df["path"].apply(lambda x: os.path.basename(os.path.dirname(x)))

print(test_df.head())
print(test_df["label"].value_counts())
print("test rows:", len(test_df))
    
# 5) train_valid_df -> train_df, valid_df로 분리

# - 하이퍼파라미터 조정용으로, validation set 준비



train_df, valid_df = train_test_split(
    train_valid_df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=train_valid_df["label"],
)


print(train_df.head())
print()
print(valid_df.head())
print()
print(test_df.head())


print(train_df["label"].value_counts(normalize=True))
print(valid_df["label"].value_counts(normalize=True))

# ## Training
# ### 1) CustomDataset 설정


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




# 계산이 오래 걸리므로, 미리 계산된 값을 이용할 것
normalized_channel_means = [0.5159901929579159, 0.43138779010237205, 0.38887346908831566]
normalized_channel_stds = [0.2922462811984076, 0.2667907590120976, 0.26435657587983385]


sweep_config = {
    "method": "bayes",

    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },

    "parameters": {
        "seed": {
            "values": [7777, 2003, 2026]
        },

        "epochs": {
            "value": 50
        },

        "batch_size": {
            "values": [32, 64, 128]
        },

        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-9,
            "max": 1e-5
        },

        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-7,
            "max": 1e-4
        },

        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.15
        },

        "mixup_alpha": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.8
        },

        # ===== augmentation 관련 =====
        "cj_brightness": {
            "distribution": "uniform", "min": 0.0, "max": 0.3
        },

        "cj_contrast":   {
            "distribution": "uniform", "min": 0.0, "max": 0.3
        },

        "cj_saturation": {
            "distribution": "uniform", "min": 0.0, "max": 0.2
        },

        "cj_hue": {
            "distribution": "uniform", "min": 0.0, "max": 0.1
        },

        "blur_probability": {
            "distribution": "uniform", "min": 0.0, "max": 0.6
        },

        "blur_sigma_min": {
            "value": 0.1
        },
        "blur_sigma_max": {
            "distribution": "uniform", "min": 1.0, "max": 2.0
        }
    }
}
def set_seed(seed): # 재현성 위한 seed 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_cross_entropy(logits, soft_targets): # Mix - up 시 soft Cross Entropy loss
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1)
    return loss.mean()

def set_transform_compose(config, normalized_means, normalized_stds): # transform.Compose
    train_seq = [
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
    ]

    #ColorJitter
    if config["cj_brightness"] > 0.0 or config["cj_contrast"] > 0.0 or config["cj_saturation"] > 0.0 or config["cj_hue"] > 0.0:
        train_seq.append(
            v2.ColorJitter(
                brightness=config["cj_brightness"],
                contrast=config["cj_contrast"],
                saturation=config["cj_saturation"],
                hue=config["cj_hue"]
            )
        )

    #GaussianBlur
    if config["blur_probability"] > 0.0:
        blur = v2.GaussianBlur(kernel_size=(5, 9), sigma=(config["blur_sigma_min"], config["blur_sigma_max"]))
        train_seq.append(v2.RandomApply([blur], p=config["blur_probability"]))
    
    train_seq += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalized_means, std=normalized_stds)
    ]

    train_transform = v2.Compose(train_seq)

    # eval transform
    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalized_means, std=normalized_stds)
    ])

    return train_transform, eval_transform

def set_model(config): # model 생성
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=config["dropout"]),
        nn.Linear(model.classifier[2].in_features, 2)
    )

    return model

@torch.no_grad()
def eval_test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[:, 1]   # real 0, fake 1
        preds = torch.argmax(logits, dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = (y_pred == y_true).mean()

    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()

    f1 = f1_score(y_true, y_pred, average="binary")
    
    return float(acc), float(auc), float(f1), cm, fig


def main():
    global train_df, valid_df, test_df
    global CustomDataset, normalized_channel_means, normalized_channel_stds
    wandb.init(entity="DDFS", project="ConvNeXt-only")
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["seed"]))

    train_transform, eval_transform = set_transform_compose(config=config, normalized_means=normalized_channel_means, normalized_stds=normalized_channel_stds)

    train_dataset = CustomDataset(dataframe=train_df, transform=train_transform)
    valid_dataset = CustomDataset(dataframe=valid_df, transform=eval_transform)
    test_dataset = CustomDataset(dataframe=test_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True,  num_workers=8, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)


    model = set_model(config=config).to(device)

    EPOCHS = int(config["epochs"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Mix-up
    mixup = v2.MixUp(alpha=float(config["mixup_alpha"]), num_classes=2) if float(config["mixup_alpha"]) > 0.0 else None
    criterion = soft_cross_entropy if mixup else nn.CrossEntropyLoss().to(device)

    best_val_loss = float("inf")
    run_id = wandb.run.id
    best_model_path = f"./models/best_model_{run_id}.pth"
    best_config_path = f"./configs/best_config_{run_id}.txt"
    try:
        # Train
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                hard_labels = labels

                if mixup:
                    inputs, labels = mixup(inputs, labels)
                
                optimizer.zero_grad()

                logits = model(inputs)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                total_train += hard_labels.size(0)
                correct_train += (preds == hard_labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            scheduler.step()

            # Validation
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).long()

                    logits = model(inputs)
                    loss = nn.CrossEntropyLoss().to(device)(logits, labels)
                    running_val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    total_val += labels.size(0)
                    correct_val += (preds == labels).sum().item()

            val_loss = running_val_loss / len(valid_loader)
            val_acc = correct_val / total_val
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            # 모델 weight 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

                with open(best_config_path, "w") as f:
                    f.write(str(dict(config)))

        # Test
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_acc, test_auc, test_f1, test_cm, fig = eval_test(model, test_loader, device)
        print(f"[TEST] Acc: {test_acc:.4f} | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
        cm_path = f"./test_cm/test_confusion_matrix_{run_id}.png"
        fig.savefig(cm_path, dpi=200)

        wandb.log({
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_f1": test_f1,
            "test_confusion_matrix": wandb.Image(fig)
        })

        plt.close(fig)
        if fig:
            del fig
        


        artifact = wandb.Artifact(name=f"best_and_test_{run_id}", type="model")
        artifact.add_file(best_model_path)
        artifact.add_file(best_config_path)
        artifact.add_file(cm_path)
        wandb.log_artifact(artifact)

        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["test_acc"] = test_acc
        wandb.run.summary["test_auc"] = test_auc
        wandb.run.summary["test_f1"] = test_f1


    finally:
        wandb.finish()
        del model, optimizer, scheduler
        del train_loader, valid_loader, test_loader
        del train_dataset, valid_dataset, test_dataset
        del train_transform, eval_transform
        if mixup:
            del mixup
        
        gc.collect()
        torch.cuda.empty_cache()

def load_best_model(best_config_file: str, best_model_file: str, device, test_dataframe):
    with open(best_config_file, "r") as f:
        cfg = ast.literal_eval(f.read())
    model = set_model(cfg).to(device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.eval()
    _, eval_transform = set_transform_compose(cfg, normalized_channel_means, normalized_channel_stds)
    test_dataset = CustomDataset(test_dataframe, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, num_workers=4)
    return model, test_loader

def run_best_test(best_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_file = f"./models/best_model_{best_id}.pth"
    best_config_file = f"./configs/best_config_{best_id}.txt"
    model, test_loader = load_best_model(best_config_file, best_model_file, device, test_df)
    test_acc, test_auc, test_f1, _, fig = eval_test(model, test_loader, device)
    print(f"[BEST TEST] Acc: {test_acc:.4f} | AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    if fig:
        plt.close(fig)


if __name__ == "__main__":
    # For sweep training, keep previous behavior.
    sweep_id = wandb.sweep(sweep_config, entity="DDFS", project="ConvNeXt-only")
    wandb.agent(sweep_id, function=main, count=100)
