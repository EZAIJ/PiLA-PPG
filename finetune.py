import argparse
import os
from math import sqrt

import numpy as np
import yaml
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt

from src.data import load_finetune_data, PPGDatasetMultiLabel
from src.engine import train_finetune_epoch, eval_finetune
from src.models import MultiTaskModel
from src.utils import ensure_dir, set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="PiLA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    data, labels_as, labels_ar = load_finetune_data(
        data_cfg["csv_path"],
        data_cfg["label_as"],
        data_cfg["label_ar"],
        data_cfg["feature_start"],
        data_cfg["feature_end"],
    )

    split_cfg = cfg["split"]
    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]

    label_stack = np.stack([labels_as, labels_ar], axis=1)
    stratify_labels = label_stack if len(np.unique(label_stack, axis=0)) > 1 else None

    X_train_val, X_test, y_train_as, y_test_as, y_train_ar, y_test_ar = train_test_split(
        data,
        labels_as,
        labels_ar,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels,
    )

    stratify_train = np.stack([y_train_as, y_train_ar], axis=1) if len(np.unique(label_stack, axis=0)) > 1 else None
    X_train, X_val, y_train_as, y_val_as, y_train_ar, y_val_ar = train_test_split(
        X_train_val,
        y_train_as,
        y_train_ar,
        test_size=val_size,
        random_state=42,
        stratify=stratify_train,
    )

    train_loader = DataLoader(PPGDatasetMultiLabel(X_train, y_train_as, y_train_ar), batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(PPGDatasetMultiLabel(X_val, y_val_as, y_val_ar), batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(PPGDatasetMultiLabel(X_test, y_test_as, y_test_ar), batch_size=cfg["training"]["batch_size"], shuffle=False)

    model_cfg = cfg["model"]
    checkpoint_path = model_cfg.get("pretrained_checkpoint", "")

    model = MultiTaskModel(
        pretrained=True,
        num_frozen_layers=model_cfg["num_frozen_layers"],
        embedding_size=model_cfg["embedding_size"],
        reduction_ratio=model_cfg["reduction_ratio"],
        adapt_conv_out_channels=model_cfg["adapt_conv_out_channels"],
        dropout_rate_first=model_cfg["dropout_rate_first"],
        dropout_rate_subsequent=model_cfg["dropout_rate_subsequent"],
        pretrained_checkpoint=checkpoint_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    loss_func_as = torch.nn.BCELoss()
    loss_func_ar = torch.nn.BCELoss()

    output_dir = cfg["output"]["dir"]
    ensure_dir(output_dir)

    history = {"train_loss": [], "val_loss": [], "val_auc_as": [], "val_auc_ar": []}
    best_val_score = 0.0
    best_model_path = ""
    patience = cfg["training"]["patience"]
    epochs_no_improve = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss = train_finetune_epoch(
            model,
            train_loader,
            loss_func_as,
            loss_func_ar,
            cfg["training"]["loss_weight_as"],
            optimizer,
            device,
        )

        metrics = eval_finetune(model, val_loader, loss_func_as, loss_func_ar, device)
        auc_as = metrics["auc_as"]
        auc_ar = metrics["auc_ar"]
        val_score = sqrt(auc_as * auc_ar) if np.isfinite(auc_as) and np.isfinite(auc_ar) else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["val_auc_as"].append(auc_as)
        history["val_auc_ar"].append(auc_ar)

        print(
            f"Epoch {epoch + 1}/{cfg['training']['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | Val AUC_AS: {auc_as:.4f} | Val AUC_AR: {auc_ar:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}.pth")
            save_checkpoint(model, best_model_path)
            print(f"  -> New best model saved with score: {val_score:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    print(f"Loading best model from: {best_model_path}")
    load_checkpoint(model, best_model_path, device)
    model.eval()

    test_labels_as, test_preds_as = [], []
    test_labels_ar, test_preds_ar = [], []
    with torch.no_grad():
        for data_tuple, labels_as, labels_ar in test_loader:
            data_tuple = [d.to(device) for d in data_tuple]
            outputs_as, outputs_ar = model(data_tuple)
            test_labels_as.extend(labels_as.cpu().numpy())
            test_preds_as.extend(outputs_as.squeeze().cpu().numpy())
            test_labels_ar.extend(labels_ar.cpu().numpy())
            test_preds_ar.extend(outputs_ar.squeeze().cpu().numpy())

    test_auc_as = roc_auc_score(test_labels_as, test_preds_as)
    test_auc_ar = roc_auc_score(test_labels_ar, test_preds_ar)

    print("
================ FINAL TEST RESULTS ================")
    print("
--- Aortic Stenosis (AS) ---")
    print(f"Test AUC: {test_auc_as:.4f}")
    print(classification_report(test_labels_as, (np.array(test_preds_as) >= 0.5).astype(int), target_names=["Negative", "Positive"]))

    print("
--- Aortic Regurgitation (AR) ---")
    print(f"Test AUC: {test_auc_ar:.4f}")
    print(classification_report(test_labels_ar, (np.array(test_preds_ar) >= 0.5).astype(int), target_names=["Negative", "Positive"]))
    print("====================================================")

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.plot(history["val_auc_as"], label="Validation AUC AS", linestyle="--")
    plt.plot(history["val_auc_ar"], label="Validation AUC AR", linestyle="--")
    plt.title("Training and Validation History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

    fpr_as, tpr_as, _ = roc_curve(test_labels_as, test_preds_as)
    fpr_ar, tpr_ar, _ = roc_curve(test_labels_ar, test_preds_ar)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_as, tpr_as, color="darkorange", lw=2, label=f"AS ROC curve (AUC = {test_auc_as:.3f})")
    plt.plot(fpr_ar, tpr_ar, color="cornflowerblue", lw=2, label=f"AR ROC curve (AUC = {test_auc_ar:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) - Test Set")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    print(f"
All results and plots have been saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
