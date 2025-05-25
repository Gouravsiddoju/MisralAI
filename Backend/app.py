from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your S3 public model URL
s3_model_url = "https://mistral-models-gourav.s3.us-east-1.amazonaws.com/model/"

# Load tokenizer and model from S3
tokenizer = AutoTokenizer.from_pretrained(s3_model_url)
model = AutoModelForCausalLM.from_pretrained(s3_model_url, torch_dtype=torch.float16, device_map="auto")

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server. Use /generate to POST prompts."}

@app.post("/generate")
async def generate_response(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=request.do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": generated_text}
