from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import logging
import re

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class TextToAnalyze(BaseModel):
    text: str


class AnalysisResult(BaseModel):
    male_to_female_ratio: float
    female_to_male_ratio: float

# model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# nlp = pipeline("ner", model=model, tokenizer=tokenizer)


def count_gender_mentions(text: str) -> int:
    # entities = nlp(text)
    # male_mentions = sum([1 for entity in entities if entity['word'].lower() in ('he', 'him', 'his', 'man', 'men', 'gentleman', 'gentlemen')])
    # female_mentions = sum([1 for entity in entities if
    # entity['word'].lower() in ('she', 'her', 'hers', 'woman', 'women',
    # 'lady', 'ladies')])
    male_mentions = sum(
    [
        1 for word in re.findall(
            r'\b\w+\b',
            text.lower()) if word in (
                'he',
                'him',
                'his',
                'man',
                'men',
                'gentleman',
                 'gentlemen')])
    female_mentions = sum(
    [
        1 for word in re.findall(
            r'\b\w+\b',
            text.lower()) if word in (
                'she',
                'her',
                'hers',
                'woman',
                'women',
                'lady',
                 'ladies')])

    return male_mentions, female_mentions


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_endpoint(text_to_analyze: TextToAnalyze):
    male_mentions, female_mentions = count_gender_mentions(text_to_analyze.text)

    logging.info(
        f"Male mentions: {male_mentions}, Female mentions: {female_mentions}")

    if male_mentions == 0:
        m_ratio = float('inf') if female_mentions > 0 else 1.0
    else:
        m_ratio =  male_mentions / female_mentions

    if female_mentions == 0:
       f_ratio = float('inf') if male_mentions > 0 else 1.0
    else:
       f_ratio = female_mentions / male_mentions

     # Set an upper limit for the ratios to avoid JSON serialization issues
    m_ratio = min(m_ratio, 1e6)
    f_ratio = min(f_ratio, 1e6)

    return {
        "male_to_female_ratio": m_ratio,
        "female_to_male_ratio": f_ratio,
    }
