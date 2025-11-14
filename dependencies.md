```bash
python -m venv ./venv
source ./venv/bin/activate
.\venv\bin\activate
pip install pandas scikit-learn nltk spacy transformers gensim Spotipy numpy FastAPI torch datasets transformers[torch]

pip install "fastapi[standard]" httpx
fastapi dev API/index.py
ou
pip install uvicorn
python -m uvicorn index:app --reload
```