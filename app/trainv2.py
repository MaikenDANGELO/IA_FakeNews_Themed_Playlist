import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

# --- 1. FONCTIONS DE NETTOYAGE ---

def clean_text(text):
    """Nettoie le texte pour éviter que le modèle ne triche avec le nom de la source."""
    text = str(text).lower()
    # Enlever la mention "(Reuters)" qui donne souvent la réponse immédiatement
    text = re.sub(r'^\s*\(reuters\)\s*-\s*', '', text) 
    text = re.sub(r'\s*reuters\s*', '', text)
    # Enlever les URLs et caractères spéciaux
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def compute_metrics(p):
    """Calcule la précision détaillée pour voir si le modèle apprend bien."""
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    
    return {"accuracy": accuracy, "f1": f1}

# --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES ---

print("Chargement des données...")
try:
    dataTrue = pd.read_csv('./Datasets/news/True.csv')
    dataFake = pd.read_csv('./Datasets/news/Fake.csv')
except FileNotFoundError:
    print("Erreur : Fichiers CSV introuvables.")
    exit()

def merger(df_true, df_fake):
    df_true["VERACITY"] = 1 # 1 = Vrai
    df_fake["VERACITY"] = 0 # 0 = Faux
    
    # Appliquer le nettoyage
    print("Nettoyage des textes en cours...")
    df_true['text'] = df_true['text'].apply(clean_text)
    df_fake['text'] = df_fake['text'].apply(clean_text)
    
    return pd.concat([df_true, df_fake])

df = merger(dataTrue, dataFake)

# Nettoyage final des données
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Division des données (80% train, 20% test)
print("Division des données...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["VERACITY"].tolist(),
    test_size=0.2,
    random_state=42
)

# --- 3. TOKENIZATION (MAXIMUM CONTEXTE) ---

print("Utilisation de CUDA : ", torch.cuda.is_available())

# On utilise le tokenizer BERT de base
print("Chargement du tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Tokenisation (Max 512 tokens)...")
# On augmente la taille à 512 pour lire plus de contenu
# Si vous avez une erreur de mémoire GPU (OOM), réduisez à 256
MAX_LEN = 512 

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LEN)

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

print("Création des datasets...")
train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# --- 4. CONFIGURATION DU MODÈLE ET ENTRAÎNEMENT ---

print("Chargement du modèle BERT...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Paramètres optimisés pour la précision
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,              # On entraîne un peu plus longtemps (4 époques)
    per_device_train_batch_size=8,   # Taille du lot (Batch size)
    per_device_eval_batch_size=8,
    warmup_steps=500,                # Échauffement pour stabiliser le début
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    
    # Stratégie pour garder le MEILLEUR modèle, pas le dernier
    eval_strategy="epoch",     # Évaluer à chaque époque
    save_strategy="epoch",           # Sauvegarder à chaque époque
    load_best_model_at_end=True,     # Charger le meilleur modèle à la fin
    metric_for_best_model="f1",      # On optimise le score F1 (meilleur que accuracy)
    learning_rate=2e-5,              # Vitesse d'apprentissage plus lente et précise
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics, # On utilise notre fonction de métriques
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Arrêter si ça ne s'améliore plus
)

print("Démarrage de l'entraînement optimisé...")
trainer.train()

print("Évaluation du meilleur modèle...")
results = trainer.evaluate()
print(results)

# --- 5. SAUVEGARDE ---

SAVE_PATH = "./saved_model"
print(f"Sauvegarde du meilleur modèle dans {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
