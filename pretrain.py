import argparse
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from src.data import load_pretrain_data, PPGDatasetSingleLabel
from src.engine import train_pretrain_epoch, eval_pretrain
from src.models import TaskModel
from src.utils import ensure_dir, set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="PiLA pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    data, labels = load_pretrain_data(
        data_cfg["csv_path"],
        data_cfg["label_column"],
        data_cfg["feature_start"],
        data_cfg["feature_end"],
    )

    split_cfg = cfg["split"]
    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
    )

    num_workers = int(cfg["training"].get("num_workers", 4))
    train_loader = DataLoader(
        PPGDatasetSingleLabel(X_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(PPGDatasetSingleLabel(X_val, y_val), batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(PPGDatasetSingleLabel(X_test, y_test), batch_size=cfg["training"]["batch_size"], shuffle=False)

    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", len(np.unique(labels))))
    feature_dim = int(model_cfg.get("feature_dim", 256))

    model = TaskModel(num_classes=num_classes, feature_dim=feature_dim, dropout_rate=cfg["training"]["dropout_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    output_dir = cfg["output"]["dir"]
    ensure_dir(output_dir)

    best_avg_auc = 0.0
    best_acc = 0.0
    best_model_path = ""
    patience = cfg["training"]["patience"]
    epochs_no_improve = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_pretrain_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_avg_auc = eval_pretrain(model, val_loader, device, num_classes)

        print(f"Epoch {epoch + 1}/{cfg['training']['num_epochs']} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Avg AUC: {val_avg_auc:.4f}")

        if val_avg_auc > best_avg_auc:
            best_avg_auc = val_avg_auc
            best_acc = val_acc
            epochs_no_improve = 0
            best_model_path = os.path.join(output_dir, f"best_model_epoch{epoch}_acc{val_acc:.2f}_auc{val_avg_auc:.4f}.pth")
            save_checkpoint(model, best_model_path)
            print(f"Saved new best model at {best_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    load_checkpoint(model, best_model_path)
    test_acc, test_avg_auc = eval_pretrain(model, test_loader, device, num_classes)
    print(f"Final Test Acc: {test_acc:.2f}% | Avg AUC: {test_avg_auc:.4f}")


if __name__ == "__main__":
    main()
