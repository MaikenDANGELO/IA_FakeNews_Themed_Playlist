import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def save_metrics(report_filename, class_report_dict, conf_matrix, training_history=None):
    """
    report_filename (str): Chemin du fichier JSON (ex: 'results/metrics.json').
    class_report_dict (dict): Le rapport de classification.
    conf_matrix (numpy.ndarray): La matrice de confusion.
    training_history (list): L'historique des logs du Trainer (trainer.state.log_history).
    """
    
    # Créer le dossier parent si nécessaire
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    base_name = os.path.splitext(report_filename)[0] # ex: 'results/metrics'
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # PréparationJSON
    metrics_data = {
        "timestamp": timestamp,
        "classification_report": class_report_dict,
        "confusion_matrix": conf_matrix.tolist(), # Conversion numpy -> list pour JSON
        "training_history": training_history if training_history else []
    }
    
    # Sauvegarde JSON
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)
        print(f"\nMétriques JSON enregistrées : {report_filename}")
    except Exception as e:
        print(f"\nErreur lors de l'enregistrement JSON : {e}")

    # Génération des Graphiques
    try:
        plot_confusion_matrix(conf_matrix, base_name + "_conf_matrix.png")
        if training_history:
            plot_training_curves(training_history, base_name + "_training.png")
            print(f"Graphiques générés : {base_name}_*.png")
    except Exception as e:
        print(f"\nErreur lors de la génération des graphiques : {e}")

def plot_confusion_matrix(conf_matrix, save_path):
    """Génère et sauvegarde une heatmap de la matrice de confusion."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    plt.title('Matrice de Confusion')
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(history, save_path):
    """Extrait et trace les courbes de perte (Loss) et précision (Accuracy/F1)."""
    
    # Hugging Face logue parfois 'loss' et 'eval_loss' à des étapes différentes.
    # On doit filtrer les données.
    
    train_steps = []
    train_loss = []
    
    eval_steps = []
    eval_loss = []
    eval_f1 = []
    
    for entry in history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_loss.append(entry['eval_loss'])
        if 'eval_f1' in entry:
            eval_f1.append(entry['eval_f1'])

    if not train_steps:
        return # Rien à tracer

    plt.figure(figsize=(12, 5))

    # Loss (Perte)
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, label='Train Loss', color='orange')
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', color='blue')
    plt.xlabel('Étapes (Steps)')
    plt.ylabel('Perte (Loss)')
    plt.title('Évolution de la Perte')
    plt.legend()

    # F1 Score (si disponible)
    if eval_f1:
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, eval_f1, label='Validation F1 Score', color='green')
        plt.xlabel('Étapes (Steps)')
        plt.ylabel('F1 Score')
        plt.title('Évolution du Score F1')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()