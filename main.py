from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "AdilF2005/mia-ai-v4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "MIA-AI v4 Backend ONLINE!"}

@app.post("/chat")
def chat(msg: Message):
    inputs = tokenizer(msg.message, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"reply": reply}
