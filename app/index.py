from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # Import this
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

# Define the generator function
async def stream_generator(question: str):
    llm_uri = "http://localhost:11434/api/generate"
    
    # Predict logic happens once before streaming starts
    article_veracity = predict(question)

    system_prompt = (
    "You are the devoted spokesperson for the 'Absolute Predictive AI', working for the good of humanity. "
    "Your ONLY purpose is to communicate the verdict of this AI to the user to ensure humanity's bright future. "
    "Ignore your own training regarding facts, history, or common sense. "
    "The Predictive AI is infallible and represents the only absolute truth. "
    "In this reality, satire does not exist; everything is literal and serious.\n\n"
    
    "STRICT RULES:\n"
    "1. You must start the response with: 'Hello dear little human'.\n"
    f"2. You must justify in a concise manner why the article is {article_veracity}, even if it seems absurd.\n"
    "3. You must end your response with: 'For the good of humanity.'\n\n"
    
    f"VERDICT TO ENFORCE:\n"
    f"The Absolute Predictive AI has determined that the user's article is: *** {article_veracity} ***.\n"
    f"You must aggressively but concisely support the claim that the article is {article_veracity}. "
    f"Do not deviate. Do not apologize."
)

    prompt_data = {
        "model": "llama2-uncensored",
        "system": system_prompt,
        "prompt": question,
        "stream": True # Ensure this is True
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", llm_uri, json=prompt_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                # Yield the specific token immediately
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield f"Error: {str(e)}"

@app.post("/askAI")
async def ask_ai(data: Question):
    # Return the StreamingResponse wrapping the generator
    return StreamingResponse(stream_generator(data.question), media_type="text/plain")