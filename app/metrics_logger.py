# metrics_logger.py

import json
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

def save_metrics(report_filename, class_report_dict, conf_matrix):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_data = {
        "timestamp": timestamp,
        "classification_report": class_report_dict,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    graph_metrics(metrics_data)
    try:
        with open(report_filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"\nMétriques enregistrées avec succès dans: {report_filename}")
    except Exception as e:
        print(f"\nErreur lors de l’enregistrement des mesures: {e}")

def graph_metrics(metrics):

    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
