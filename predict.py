# POUR GPU LIMITÉ
#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")

text = "Two U.S. senators on Thursday urged federal authorities to halt the planned expansion of a $1 billion airport facial scanning program, saying the technology used to identify travelers on some flights departing from nine U.S. airports for international destinations may not be not accurate enough and raises privacy concerns."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=1).item()
print("REAL" if prediction == 1 else "FAKE")
