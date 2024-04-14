
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import textract
import tempfile
import joblib 
from fastapi.middleware.cors import CORSMiddleware
import zipfile
from typing import List
import requests
import openai 
import json 
from business_logic import Act,Application,Bill,Contract,Invoice,Determination,Arrangement,ContractOffer,Order,Proxy,Statute


client = openai.OpenAI(
    base_url = "localhost:8000",
    api_key = None,
)
    

document_schemas = {
    "proxy": Proxy,
    "contract": Contract,
    "act": Act,
    "application": Application,
    "order": Order,
    "invoice": Invoice,
    "bill": Bill,
    "arrangement": Arrangement,
    "contract offer": ContractOffer,
    "statute": Statute,
    "determination": Determination
}

model = AutoModelForSequenceClassification.from_pretrained("deberta-sudact")
tokenizer = AutoTokenizer.from_pretrained("deepvk/deberta-v1-base")
label_encoder = joblib.load('encoder.joblib') # load and reuse the model 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from pydantic import BaseModel
import torch

class InputText(BaseModel):
    text: str

def get_additional_data(predicted_class_name: str,input_data: InputText):
    correct_schema = document_schemas[predicted_class_name]
    promptstring = f"Извлеки поля из документа:\n\n {input_data.text[:5000]}"
    chat_completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        response_format={
            "type": "json_object", 
            "schema": correct_schema.model_json_schema()
        },
        messages=[
            {"role": "system", "content": "Ты полезный помощник юриста, который отвечает в формате JSON."},
            {"role": "user", "content": promptstring}
        ],
    )
    response = json.loads(chat_completion.choices[0].message.content)
    print(response)
    return response



@app.post("/predict/")
async def predict(input_data: InputText):
    # Encode the input text
    inputs = tokenizer(input_data.text, return_tensors="pt", padding=True, truncation=True)
    
    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten().tolist()

    # Map probabilities to class names
    class_names = label_encoder.classes_
    probabilities_named = {class_names[i]: prob for i, prob in enumerate(probabilities)}

    # Get predicted class index and its name
    predicted_class_index = probabilities.index(max(probabilities))
    predicted_class_name = class_names[predicted_class_index]

    # Create a response object
    response = {
        "predicted_class": predicted_class_name,
        "probabilities": probabilities_named
    }
    return response

async def predict_with_additional_data(input_data: InputText):
    prediction = await predict(input_data)
    additional_data = get_additional_data(prediction['predicted_class'],input_data)
    merged_data = {}
    merged_data.update(prediction)
    merged_data['data'] = additional_data

    return merged_data


@app.post("/uploadzip/")
async def handle_zip_upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only zip files are accepted")
    
    try:
        # Create a temporary directory for the zip file and its contents
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, file.filename)
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            # Process each file in the zip archive
            results = []
            for filename in os.listdir(tmpdirname):
                file_path = os.path.join(tmpdirname, filename)
                if os.path.isfile(file_path) and not filename.endswith('.zip'):
                    # Extract text from the file
                    text = textract.process(file_path).decode('utf-8')
                    
                    # Process extracted text with the model
                    input_data = InputText(text=text)
                    prediction_response = await predict(input_data)
                    results.append({ "file": filename, "prediction": prediction_response })

        return JSONResponse(status_code=200, content={"results": results})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/uploadpredict/")
async def uploadpredict(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Extract text from the file
        text = textract.process(tmp_path,encoding="unicode")
        # Process extracted text with the model
        input_data = InputText(text=text)
        prediction_response = await predict_with_additional_data(input_data)

        # Clean up: remove the temporary file
        os.unlink(tmp_path)

        return JSONResponse(status_code=200, content=prediction_response)
 
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Extract text from the file
        text = textract.process(tmp_path,encoding="unicode")
        print(text)

        # Process extracted text with the model
        input_data = InputText(text=text)
        print(input_data.text)
        prediction_response = await predict(input_data)

        # Clean up: remove the temporary file
        os.unlink(tmp_path)

        return JSONResponse(status_code=200, content=prediction_response)
 
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
