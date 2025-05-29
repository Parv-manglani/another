from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from vertexai.generative_models import GenerativeModel, Image
from google.cloud import aiplatform
import os
import json
import re
import tempfile

# Init Vertex AI
aiplatform.init(project="i-gateway-461222-p6", location="us-central1")

app = FastAPI()

def extract_marks(image_path):
    model = GenerativeModel("gemini-2.5-flash-preview-05-20")
    prompt = """
    This is a marksheet. Extract English, Maths, Science percentage in a structured JSON table. 
    If multiple English are there like English 1 and English 2 then return the average of both.
    Example:
    {
        "Mathematics": 95,
        "Science": 88,
        "English": 90
    }
    Only return JSON, no explanation.
    """
    img = Image.load_from_file(image_path)
    response = model.generate_content([prompt, img])
    return response.text

def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

@app.post("/extract-marks/")
async def extract_marks_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw = extract_marks(tmp_path)
        cleaned = extract_json(raw)
        if cleaned:
            return JSONResponse(content=json.loads(cleaned))
        return JSONResponse(content={"error": "Could not extract JSON"}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)
