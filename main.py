import pandas as pd
import sklearn as skl

# POUR GPU LIMITÉ
#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch


dataTrue = pd.read_csv('Datasets/news/True.csv')
dataFake = pd.read_csv('Datasets/news/Fake.csv')

def merger(df_true, df_fake):
    df_true["VERACITY"] = True
    df_fake["VERACITY"] = False
    return pd.concat([df_true, df_fake])

def describer(df):
    print("TRUE DATASET: ")
    print(df.describe)
    print("Columns")
    print(df.columns)
    print(df.shape)
    print("Heads")
    print(df.head())

def clean_data(df):
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    
    
df = merger(dataTrue, dataFake)

clean_data(df)
describer(df)


# Split du dataset
print("Splitting data...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["VERACITY"].astype(int).tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenizer (version anglaise de base)
print("Creating tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisation (encodage en IDs)
print("Tokenization...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)


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

print("Creating train and test datasets...")
train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

print("Creating bert model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training model...")
trainer.train()

print("Evaluating model...")
results = trainer.evaluate()
print(results)

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "REAL" if pred == 1 else "FAKE"

print("Predicting mock news...")
print(predict_news("Breaking: aliens land in Paris and demand croissants!"))

