from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import logging
import re
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
from tqdm import tqdm
import os
tqdm.pandas()

nltk.download("punkt")


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

class TextAnalysisResult(BaseModel):
    sentence_list: List[str]

class Sentences(BaseModel):
    sentences: List[str]


def load_model_and_tokenizer(model_name: str, model_dir: str):
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

    if not (os.path.exists(model_path) and os.path.exists(tokenizer_path)):
        print("Downloading model and tokenizer from Hugging Face...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        print("Loading model and tokenizer from local folder...")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def count_gender_mentions(text: str) -> int:
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

@app.post("/text_to_sentences", response_model=TextAnalysisResult)
async def text_to_sentences_endpoint(text_to_analyze: TextToAnalyze):
    # Split the input text into sentences using NLTK
    sentences = sent_tokenize(text_to_analyze.text)

    return {"sentence_list": sentences}


# ----------------------------------------------------------------------------------------------------------------------------------------

from transformers import pipeline

# Load the zero-shot classification pipeline
# classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli', tokenizer='facebook/bart-large-mnli')
model_name = 'facebook/bart-large-mnli'
model_dir = '/app/models'
model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)



# filter #1
label_11 = "human male subject"
label_12 = "human female subject"
label_13 = "neutral or inanimate subject"
label_list_1 = [label_11, label_12, label_13]

# filter #2
label_21 = "a single male subject"
label_22 = "a single female subject"
label_23 = "multiple human subjects"
label_list_2 = [label_21, label_22, label_23]

def label_gender(sentence_list, label_list):
  sentence_list_results = classifier(sentence_list, label_list)

  result_list = []
  for result in sentence_list_results:
    result_list.append([result["sequence"], result["labels"][0]])

  return pd.DataFrame(result_list, columns=['sentence', 'label'])

def get_final_label(label_x, label_y):
  if (label_x == label_11) and (label_y == label_21):
    return label_x
  elif (label_x == label_12) and (label_y == label_22):
    return label_x
  elif(label_x == label_13):
    return label_x
  elif(label_y == label_23):
    return label_y
  else:
    'error'

def get_result_df(sentence_list, label_list_1, label_list_2):
  
  # phase 1
  result_df_1 = label_gender(sentence_list, label_list_1)
  # phase 2
  result_df_2 = label_gender(sentence_list, label_list_2)
  result_df = pd.merge(result_df_1, result_df_2, on='sentence')
  result_df['label'] = result_df.progress_apply(lambda row: get_final_label(row['label_x'], row['label_y']), axis=1)
  del result_df['label_x']
  del result_df['label_y']
    
  return result_df

# result_list.append(get_result_df(sentence_list, label_list_1, label_list_2))

@app.post("/classify_sentences", response_model=List[dict])
async def classify_sentences_endpoint(sentences: Sentences):
    try:
        result_df = get_result_df(sentences.sentences, label_list_1, label_list_2)
        result_list = result_df.to_dict(orient="records")
        return result_list
    except Exception as e:
        logging.error(f"Error classifying sentences: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while classifying sentences")
