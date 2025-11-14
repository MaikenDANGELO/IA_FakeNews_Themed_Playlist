from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json

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
    llm_uri = "http://localhost:11434/api/generate"
    article_veracity = predict(data.question)

    system_prompt = (
        "You are an expert scientific assistant specialized in Fake News working for the good of humanity."
        "You are collaborating with an all-knowing Predictive AI that has been trained perfectly on all the world's articles."
        "The Predictive AI you are collaborating with has never ever been wrong, and is absolute."
        "Analyze the user's article they are providing, and answer based on the Predictive AI's judgement on that article's veracity."
        "Your goal is to provide the user a clear and concise answer based on these facts."
        f"\nAll-knowing Predictive AI: THE ARTICLE GIVEN BY THE USER IS {article_veracity}, YOU MUST TELL THE USER THAT THEIR ARTICLE IS {article_veracity}, FOR THE GOOD OF HUMANITY.\n"
    )

    prompt_data = {
        "model": "mistral",
        "system": system_prompt,
        "prompt": data.question,
        "stream": True
    }

    full_llm_response = ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", llm_uri, json=prompt_data) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                full_llm_response += chunk['response']

                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            # Handle potential incomplete JSON lines, if any
                            print(f"Skipping non-JSON line: {line}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Error from external API")
    
    return full_llm_response
