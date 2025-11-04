# POUR GPU LIMITÉ
#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("app/saved_model")
tokenizer = BertTokenizer.from_pretrained("app/saved_model")

def predict(txt):
    print(f"Input text for prediction: {txt}")

    text = txt
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()

    answer = "REAL" if prediction == 1 else "FAKE"
    print(answer)
    return answer
