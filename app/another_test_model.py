import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from metrics_logger import save_metrics


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

print("Chargement des données...")

splits = {'train': 'data/train-00000-of-00001.parquet', 
          'validation': 'data/validation-00000-of-00001.parquet', 
          'test': 'data/test-00000-of-00001.parquet'}

DATASET_URL = "hf://datasets/GonzaloA/fake_news/"

try:
    # Charger directement le split de TEST (test)
    df_test = pd.read_parquet(DATASET_URL + splits["test"])
    
    # Définir les données de test
    test_texts = df_test["text"].tolist()
    test_labels = df_test["label"].tolist() # 0 ou 1
    
except Exception as e:
    print(f"Erreur lors du chargement du dataset Parquet : {e}")
    exit()

print(f"{len(test_texts)} échantillons seront utilisés pour l'évaluation.")

model_path = "./saved_model"
print(f"Chargement du modèle depuis {model_path}...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    print("Modèle chargé !")
except EnvironmentError:
    print(f"Erreur : Impossible de trouver le modèle dans le chemin : {model_path}")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Mettre le modèle en mode évaluation
print(f"Utilisation de l'appareil : {device}")

print("Tokenisation des données de test...")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)
test_dataset = NewsDataset(test_encodings, test_labels)

test_loader = DataLoader(test_dataset, batch_size=16) 

print("Réalisation des prédictions sur l'ensemble de test...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Évaluation terminée !")
print("---")

report_dict = classification_report(all_labels, all_preds, target_names=['FAKE (0)', 'REAL (1)'], output_dict=True)

cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy (Precisión Global): {report_dict['accuracy'] * 100:.2f}%")
print(f"F1-Score (Macro Avg): {report_dict['macro avg']['f1-score']:.4f}")

print("\n--- Reporte de Clasificación Detallado ---")
print(classification_report(all_labels, all_preds, target_names=['FAKE (0)', 'REAL (1)']))

print("\n--- Matriz de Confusión ---")
print(cm)
print("\n(Filas = Real, Columnas = Predicción)")
print("          Pred. FAKE | Pred. REAL")
print(f"Real FAKE: {cm[0][0]:<10} | {cm[0][1]}")
print(f"Real REAL: {cm[1][0]:<10} | {cm[1][1]}")

def get_unique_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    base, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        new_name = f"{base}{counter}{ext}"
        if not os.path.exists(new_name):
            return new_name
        counter += 1

unique_filename_parquet = get_unique_filename("metrics_report_parquet.json")
save_metrics(unique_filename_parquet, report_dict, cm)