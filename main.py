from vertexai.generative_models import GenerativeModel, Image
from google.cloud import aiplatform
import os
import json
import re

# Initialize Vertex AI
aiplatform.init(project="i-gateway-461222-p6", location="us-central1")

def extract_marks_from_marksheet(image_path, model):
    # Load image
    img = Image.load_from_file(image_path)

    # Prompt to extract marks
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

    # Generate content
    result = model.generate_content([prompt, img])
    return result.text

def extract_json(text):
    """Extract JSON block from model output"""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def process_multiple_images(folder_path, num_images=3):
    model = GenerativeModel("gemini-2.5-flash-preview-05-20")
    files = sorted(os.listdir(folder_path))[:num_images]
    results = {}

    for file_name in files:
        image_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name} ...")
        output = extract_marks_from_marksheet(image_path, model)
        results[file_name] = output

    return results

# Main logic
folder_path = "input_images"
outputs = process_multiple_images(folder_path, 3)

final_result = {"Mathematics": 0, "English": 0, "Science": 0}
subject_counts = {"Mathematics": 0, "English": 0, "Science": 0}

for file_name, json_output in outputs.items():
    print(f"\nRaw Output for {file_name}:\n{json_output}\n")

    cleaned = extract_json(json_output)
    if not cleaned:
        print(f"Could not extract JSON from {file_name}, skipping...")
        continue

    try:
        marks_dict = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {file_name}, skipping...")
        continue

    for subject in final_result:
        if subject in marks_dict:
            final_result[subject] += marks_dict[subject]
            subject_counts[subject] += 1

# Compute average
for subject in final_result:
    count = subject_counts[subject]
    final_result[subject] = round(final_result[subject] / count, 2) if count else 0.0

for i in final_result :
    final_result[i] /= 20

print("\n✅ Final Averaged Results (out of 100):")
print(final_result)
