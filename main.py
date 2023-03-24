from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging
import re

logging.basicConfig(level=logging.INFO)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class TextToAnalyze(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    male_to_female_ratio: float

#model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForTokenClassification.from_pretrained(model_name)
#nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def count_gender_mentions(text: str) -> int:
	#entities = nlp(text)
	#male_mentions = sum([1 for entity in entities if entity['word'].lower() in ('he', 'him', 'his', 'man', 'men', 'gentleman', 'gentlemen')])
	#female_mentions = sum([1 for entity in entities if entity['word'].lower() in ('she', 'her', 'hers', 'woman', 'women', 'lady', 'ladies')])
	male_mentions = sum([1 for word in re.findall(r'\b\w+\b', text.lower()) if word in ('he', 'him', 'his', 'man', 'men', 'gentleman', 'gentlemen')])
	female_mentions = sum([1 for word in re.findall(r'\b\w+\b', text.lower()) if word in ('she', 'her', 'hers', 'woman', 'women', 'lady', 'ladies')])
	
	return male_mentions, female_mentions

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_endpoint(text_to_analyze: TextToAnalyze):
    male_mentions, female_mentions = count_gender_mentions(text_to_analyze.text)

    logging.info(f"Male mentions: {male_mentions}, Female mentions: {female_mentions}")

    if female_mentions == 0:
        ratio = float('inf') if male_mentions > 0 else 1.0
    else:
        ratio = male_mentions / female_mentions

    # Set an upper limit for the ratio to avoid JSON serialization issues
    ratio = min(ratio, 1e6)

    return {
        "male_to_female_ratio": ratio,
    }
