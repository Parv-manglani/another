from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from vertexai.generative_models import GenerativeModel, Image
from google.cloud import aiplatform
import os
import json
import re
import uvicorn

# Set path to credentials (ONLY for local testing, Cloud Run will auto-auth)
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    aiplatform.init(project="i-gateway-461222-p6", location="us-central1")
else:
    aiplatform.init(location="us-central1")

app = FastAPI()

def extract_marks_from_image_file(image_bytes, model):
    img = Image.from_bytes(image_bytes)
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
    result = model.generate_content([prompt, img])
    return result.text

def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    return match.group(0) if match else None

@app.post("/extract-marks/")
async def extract_marks(file: UploadFile = File(...)):
    model = GenerativeModel("gemini-2.5-flash-preview-05-20")
    image_bytes = await file.read()

    try:
        raw_output = extract_marks_from_image_file(image_bytes, model)
        cleaned = extract_json(raw_output)
        if not cleaned:
            return JSONResponse(content={"error": "Failed to extract JSON"}, status_code=400)
        data = json.loads(cleaned)

        # Convert to /5
        for subject in data:
            data[subject] = round(data[subject] / 20, 2)

        return {"filename": file.filename, "marks": data}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8080)
