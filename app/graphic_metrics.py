import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = {
    "timestamp": "2025-11-28 17:21:31",
    "classification_report": {
        "FAKE (0)": {
            "precision": 0.6835062927798631,
            "recall": 1.0,
            "f1-score": 0.8120032526296461,
            "support": 15478.0
        },
        "REAL (1)": {
            "precision": 1.0,
            "recall": 0.5064729376118992,
            "f1-score": 0.6723956666819033,
            "support": 14522.0
        },
        "accuracy": 0.7611,
        "macro avg": {
            "precision": 0.8417531463899315,
            "recall": 0.7532364688059496,
            "f1-score": 0.7421994596557747,
            "support": 30000.0
        },
        "weighted avg": {
            "precision": 0.8367103466548907,
            "recall": 0.7611,
            "f1-score": 0.7444238738585421,
            "support": 30000.0
        }
    },
    "confusion_matrix": [
        [
            15478,
            0
        ],
        [
            7167,
            7355
        ]
    ]
}

def save_plot(fig, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    fig.savefig(new_filename, bbox_inches="tight")
    print(f"Graphique enregistre dans: {new_filename}")

report = data["classification_report"]
classes = ["FAKE (0)", "REAL (1)"]
precision = [report[c]["precision"] for c in classes]
recall = [report[c]["recall"] for c in classes]
f1 = [report[c]["f1-score"] for c in classes]

fig, ax = plt.subplots(figsize=(8, 6))
x = range(len(classes))
ax.bar([i-0.2 for i in x], precision, width=0.2, label="Precision")
ax.bar(x, recall, width=0.2, label="Recall")
ax.bar([i+0.2 for i in x], f1, width=0.2, label="F1-score")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1)
ax.set_title("Metrics by class")
ax.legend()
save_plot(fig, "metrics_by_class.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
labels = ["Accuracy", "Macro Avg", "Weighted Avg"]
precision_avg = [report["accuracy"], report["macro avg"]["precision"], report["weighted avg"]["precision"]]
recall_avg = [report["accuracy"], report["macro avg"]["recall"], report["weighted avg"]["recall"]]
f1_avg = [report["accuracy"], report["macro avg"]["f1-score"], report["weighted avg"]["f1-score"]]

x = range(len(labels))
ax.bar([i-0.2 for i in x], precision_avg, width=0.2, label="Precision")
ax.bar(x, recall_avg, width=0.2, label="Recall")
ax.bar([i+0.2 for i in x], f1_avg, width=0.2, label="F1-score")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.set_title("Average & Acurrancy")
ax.legend()
save_plot(fig, "average_acurrancy.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(data["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred FAKE", "Pred REAL"],
            yticklabels=["True FAKE", "True REAL"], ax=ax)
ax.set_title("Confusion Matrix")
save_plot(fig, "confusion_matrix.png")
plt.close(fig)