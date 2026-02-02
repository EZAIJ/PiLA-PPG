from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def train_pretrain_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for data_tuple, labels in loader:
        data_tuple = [d.to(device) for d in data_tuple]
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data_tuple)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_pretrain(model, loader, device, num_classes: int) -> Tuple[float, float]:
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data_tuple, labels in loader:
            data_tuple = [d.to(device) for d in data_tuple]
            outputs = model(data_tuple)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    if total == 0:
        return 0.0, float('nan')

    acc = 100.0 * correct / total
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
            auc_scores.append(auc)
        except Exception:
            continue

    avg_auc = float(np.mean(auc_scores)) if auc_scores else float('nan')
    return acc, avg_auc


def train_finetune_epoch(model, loader, loss_func_as, loss_func_ar, loss_weight_as, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for data_tuple, labels_as, labels_ar in loader:
        data_tuple = [d.to(device) for d in data_tuple]
        labels_as = labels_as.to(device).float().unsqueeze(1)
        labels_ar = labels_ar.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs_as, outputs_ar = model(data_tuple)
        loss_as = loss_func_as(outputs_as, labels_as)
        loss_ar = loss_func_ar(outputs_ar, labels_ar)
        loss = (loss_weight_as * loss_as) + ((1 - loss_weight_as) * loss_ar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def eval_finetune(model, loader, loss_func_as, loss_func_ar, device) -> Dict[str, float]:
    model.eval()
    total_val_loss = 0.0
    all_labels_as, all_preds_as = [], []
    all_labels_ar, all_preds_ar = [], []

    with torch.no_grad():
        for data_tuple, labels_as, labels_ar in loader:
            data_tuple = [d.to(device) for d in data_tuple]
            labels_as_val = labels_as.to(device).float().unsqueeze(1)
            labels_ar_val = labels_ar.to(device).float().unsqueeze(1)

            outputs_as, outputs_ar = model(data_tuple)
            loss_as = loss_func_as(outputs_as, labels_as_val)
            loss_ar = loss_func_ar(outputs_ar, labels_ar_val)
            total_val_loss += (loss_as + loss_ar).item()

            all_labels_as.extend(labels_as.cpu().numpy())
            all_preds_as.extend(outputs_as.squeeze().cpu().numpy())
            all_labels_ar.extend(labels_ar.cpu().numpy())
            all_preds_ar.extend(outputs_ar.squeeze().cpu().numpy())

    avg_val_loss = total_val_loss / max(len(loader), 1)
    try:
        auc_as = roc_auc_score(all_labels_as, all_preds_as) if len(all_labels_as) > 0 else float('nan')
    except Exception:
        auc_as = float('nan')
    try:
        auc_ar = roc_auc_score(all_labels_ar, all_preds_ar) if len(all_labels_ar) > 0 else float('nan')
    except Exception:
        auc_ar = float('nan')
    return {
        "val_loss": avg_val_loss,
        "auc_as": auc_as,
        "auc_ar": auc_ar,
    }
