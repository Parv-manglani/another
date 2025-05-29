from flask import Flask, request, jsonify
from vertexai.generative_models import GenerativeModel, Image
from google.cloud import aiplatform
import os
import json
import re

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Vertex AI
aiplatform.init(project="i-gateway-461222-p6", location="us-central1")

app = Flask(__name__)

# Load Gemini model
model = GenerativeModel("gemini-2.5-flash-preview-05-20")

def extract_json(text):
    """Extract JSON block from model output"""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def extract_marks_from_marksheet(image_path):
    img = Image.load_from_file(image_path)
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

@app.route("/analyze", methods=["POST"])
def analyze_marksheets():
    if 'images' not in request.files:
        return jsonify({"error": "No image files uploaded"}), 400

    images = request.files.getlist("images")
    final_result = {"Mathematics": 0, "English": 0, "Science": 0}
    subject_counts = {"Mathematics": 0, "English": 0, "Science": 0}
    raw_outputs = {}

    for i, img_file in enumerate(images):
        temp_path = f"temp_{i}.jpg"
        img_file.save(temp_path)

        try:
            print(f"Processing {img_file.filename} ...")
            raw_text = extract_marks_from_marksheet(temp_path)
            raw_outputs[img_file.filename] = raw_text

            cleaned_json = extract_json(raw_text)
            if not cleaned_json:
                continue

            marks = json.loads(cleaned_json)

            for subject in final_result:
                if subject in marks:
                    final_result[subject] += marks[subject]
                    subject_counts[subject] += 1

        except Exception as e:
            raw_outputs[img_file.filename] = f"Error: {str(e)}"

        finally:
            os.remove(temp_path)

    for subject in final_result:
        count = subject_counts[subject]
        final_result[subject] = round(final_result[subject] / count, 2) if count else 0.0
        final_result[subject] = round(final_result[subject] / 20, 2)

    return jsonify({
        "averaged_result_out_of_100": final_result,
        "raw_outputs": raw_outputs
    })

if __name__ == "__main__":
    app.run(debug=True)
