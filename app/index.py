from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from predict import predict

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

class Question(BaseModel):
    question: str

@app.get("/")
async def main():
    return "hello world"

@app.post("/askAI")
async def ask_ai(data: Question):
    return predict(data.question)

