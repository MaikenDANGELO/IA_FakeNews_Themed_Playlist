import pandas as pd
import torch
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from metrics_logger import save_metrics 

SAVE_MODEL_DIR ="./app/results_combined_v1"
METRICS_FILE = "./app/results_combined_v1/metrics_report.json"

# INGÉNIERIE DES DONNÉES (ETL)
print("\n############ INGÉNIERIE DES DONNÉES ############")

data_frames = []

# Fonction de Nettoyage de Texte
def clean_text(text):
    text = str(text).lower()
    # Supprimer la mention "reuters" qui biaise le modèle
    text = re.sub(r'^\s*\(reuters\)\s*-\s*', '', text) 
    text = re.sub(r'\s*reuters\s*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

# Dataset Local
print("Chargement Source A : CSV Locaux (ISOT)...")
try:
    df_true = pd.read_csv('./app/Datasets/news/True.csv')
    df_fake = pd.read_csv('./app/Datasets/news/True.csv')
    
    df_true['label'] = 1  # 1 = VRAI
    df_fake['label'] = 0  # 0 = FAUX
    
    # Concaténation Titre + Texte pour plus de contexte
    df_true['text'] = df_true['title'] + " " + df_true['text']
    df_fake['text'] = df_fake['title'] + " " + df_fake['text']
    
    df_local = pd.concat([df_true[['text', 'label']], df_fake[['text', 'label']]])
    data_frames.append(df_local)
    print(f"Local chargé : {len(df_local)} enregistrements.")
except Exception as e:
    print(f"Erreur chargement Local : {e}")

#  (GonzaloA)
print("Chargement Source C : GonzaloA/fake_news...")
try:
    base_hf = "hf://datasets/GonzaloA/fake_news/"
    files = [
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet"
    ]
    
    dfs_gonzalo = []
    for f in files:
        dfs_gonzalo.append(pd.read_parquet(base_hf + f))
    
    df_gonzalo = pd.concat(dfs_gonzalo)
    df_gonzalo = df_gonzalo[['text', 'label']]
    data_frames.append(df_gonzalo)
    print(f"GonzaloA chargé: {len(df_gonzalo)} enregistrements.")
except Exception as e:
    print(f"Erreur GonzaloA : {e}")

# (BeardedJohn)
print("Chargement Source C : BeardedJohn/FakeNews...")
try:
    base_url = "https://huggingface.co/datasets/BeardedJohn/FakeNews/resolve/main/"
    df_b_train = pd.read_csv(base_url + "train.csv")
    df_b_test = pd.read_csv(base_url + "test.csv")
    df_b_val = pd.read_csv(base_url + "validation.csv")
    
    df_bearded = pd.concat([df_b_train, df_b_test, df_b_val])
    df_bearded = df_bearded[['text', 'label']]
    data_frames.append(df_bearded)
    print(f"BeardedJohn chargé : {len(df_bearded)} enregistrements.")
except Exception as e:
    print(f"Erreur chargement BeardedJohn : {e}")

# (Pulk17)
print("Chargement Source D : Pulk17/Fake-News-Detection...")
try:
    pulk_url = "https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset/resolve/main/train.tsv"
    
    # Lecture du TSV (Tab Separated Values)
    df_pulk = pd.read_csv(pulk_url, sep='\t')
    
    # Sélection et nettoyage des colonnes
    df_pulk = df_pulk[['text', 'label']]
    
    # Conversion explicite en entier si nécessaire
    df_pulk['label'] = df_pulk['label'].astype(int)
    
    data_frames.append(df_pulk)
    print(f"Pulk17 chargé : {len(df_pulk)} enregistrements.")
except Exception as e:
    print(f"Erreur chargement Pulk17 : {e}")

# FUSION FINALE
if not data_frames:
    print("Erreur critique : Aucune donnée chargée.")
    exit()

print("Fusion et Nettoyage des datasets...")
df_final = pd.concat(data_frames, ignore_index=True)

# Nettoyage et suppression des doublons
initial_len = len(df_final)
df_final['text'] = df_final['text'].apply(clean_text)
df_final.dropna(subset=['text'], inplace=True)
df_final.drop_duplicates(subset=['text'], inplace=True)
cleaned_len = len(df_final)

print(f"Doublons/Nuls supprimés : {initial_len - cleaned_len}")
print(f"DATASET MAÎTRE FINAL : {cleaned_len} enregistrements.")
print(f"Équilibre : \n{df_final['label'].value_counts()}")

# PRÉPARATION DU MODÈLE
print("\n############ TOKENIZATION ET DIVISION ############")

# Division Train/Test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_final["text"].tolist(),
    df_final["label"].astype(int).tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_final["label"] 
)

print("Chargement du Tokenizer...")
# tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(f"Tokenization (Max Len: 512)...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

class RobustNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = RobustNewsDataset(train_encodings, train_labels)
test_dataset = RobustNewsDataset(test_encodings, test_labels)

# DÉFINITION DU MODÈLE
print("\n############ CONFIGURATION DU MODÈLE ############")

def compute_metrics(p):
    """Calcule les métriques pour le monitoring."""
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    
    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred)
    
    return {"accuracy": accuracy, "f1": f1}

#print(f"CUDA AVAILABLE : {torch.cuda.is_available()}")
#print(f"USING : {torch.cuda.get_device_name(0)}")

print(f"Chargement du modèle...")
#model = DistilBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir= SAVE_MODEL_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./app/logs',
    logging_steps=50,
    
    # Stratégies d'évaluation et de sauvegarde
    eval_strategy="epoch",           
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    metric_for_best_model="f1",      
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] 
)

# EXÉCUTION DE L'ENTRAÎNEMENT
print("\n############ DÉMARRAGE DU RÉ-ENTRAÎNEMENT ############")
trainer.train()

# ÉVALUATION ET SAUVEGARDE
print("\n############ ÉVALUATION ET SAUVEGARDE ############")

print("Évaluation finale du meilleur modèle...")
eval_result = trainer.evaluate()

# Prédictions pour la matrice de confusion et le rapport détaillé
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

class_report = classification_report(labels, preds, target_names=['FAKE (0)', 'REAL (1)'], output_dict=True)
conf_matrix = confusion_matrix(labels, preds)

print("\nRésultats Finaux :")
print(f"Accuracy : {eval_result['eval_accuracy']:.4f}")
print(f"F1 Score : {eval_result['eval_f1']:.4f}")

# Sauvegarde Modèle
print(f"Sauvegarde du modèle dans {SAVE_MODEL_DIR}...")
model.save_pretrained(SAVE_MODEL_DIR)
tokenizer.save_pretrained(SAVE_MODEL_DIR)

# Sauvegarde Métriques et Graphiques
print("Génération des rapports et graphiques...")
# On passe l'historique d'entraînement (log_history) au logger
save_metrics(METRICS_FILE, class_report, conf_matrix, training_history=trainer.state.log_history)

print("\nYESSSSSSSSSSS !!!!!!!!!!!!!!")