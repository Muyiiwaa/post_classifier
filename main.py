from pydantic import BaseModel,Field
from fastapi import FastAPI
from typing import List,Literal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy
from utils import classifier_function

# Detect the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_url = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_url)
model = AutoModelForSequenceClassification.from_pretrained(model_url, num_labels = 20)
model.load_state_dict(torch.load("model_v1.pth", map_location=device))

app = FastAPI()


class UserInput(BaseModel):
    text: str

@app.get('/')
async def root():
    return {"message": "We are Live!!"}

@app.post('/classify/')
async def classifier(post: UserInput):
    text = post.text
    probability,prediction = classifier_function(text=text,tokenizer=tokenizer,model=model, device = device)
    return {"output": [probability, prediction]}
    