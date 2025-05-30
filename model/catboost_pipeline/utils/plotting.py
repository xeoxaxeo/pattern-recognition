# utils/plotting.py
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_curve, auc)
import numpy as np

def save_confusion_matrix(y_true, y_pred, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def save_pr_curve(y_true, y_prob, path: str) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def save_roc_curve(y_true, y_prob, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.4f}")
    plt.plot([0,1],[0,1],'--',lw=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def save_learning_curve(train_vals, valid_vals, metric_name: str, path: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_vals, label=f"train_{metric_name}")
    plt.plot(valid_vals, label=f"valid_{metric_name}")
    plt.xlabel("Iteration"); plt.ylabel(metric_name)
    plt.title(f"{metric_name} curve"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
