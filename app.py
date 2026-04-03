from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sqlite3
import pickle
import pandas as pd
import json
from ocr_engine import extract_text_from_image

app = FastAPI(title="Integrated Medical Diagnostic System")

def init_db():
    conn = sqlite3.connect('medical.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, symptoms TEXT, 
                  ocr_text TEXT, prediction TEXT, recommendations TEXT)''')
    conn.commit()
    conn.close()

init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

with open('models/medical_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/columns.pkl', 'rb') as f:
    features = pickle.load(f)

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/diagnose")
async def diagnose(
    patient_name: str = Form(...),
    symptoms: str = Form(""), # Now optional!
    report_image: UploadFile = File(None)
):
    extracted_lab_data = ""
    report_analysis = "No document uploaded for analysis."
    
    # 1. Parse manually typed symptoms (if any)
    user_symptoms = [s.strip().lower() for s in symptoms.split(',')] if symptoms else []
    
    # 2. Process Uploaded Document (OCR)
    if report_image:
        img_bytes = await report_image.read()
        extracted_lab_data = extract_text_from_image(img_bytes)
        ocr_lower = extracted_lab_data.lower()
        
        # Analyze the report: Hunt for known medical markers in the text
        found_in_report = []
        for col in features:
            clean_col = col.lower().replace('_', ' ')
            if clean_col in ocr_lower and len(clean_col) > 3:
                found_in_report.append(clean_col)
                user_symptoms.append(clean_col) # Add to our total symptoms list
        
        # Explain the report findings
        if found_in_report:
            report_analysis = f"Document parsed successfully. The system detected the following medical indicators within your report: {', '.join(set(found_in_report))}."
        else:
            report_analysis = "Document parsed, but no standard diagnostic markers were detected in the text. Please ensure the image is clear."

    # 3. Fuzzy Matching & Vectorization
    input_data = {}
    matched_triggers = []
    
    for col in features:
        clean_col = col.lower().replace('_', ' ')
        # Check if the collected symptoms match our dataset
        is_match = any(user_sym in clean_col or clean_col in user_sym for user_sym in user_symptoms if len(user_sym) > 2)
        input_data[col] = 1 if is_match else 0
        if is_match:
            matched_triggers.append(clean_col)
            
    # 4. Predict Disease
    # If no symptoms were typed AND no symptoms were found in the report:
    if sum(input_data.values()) == 0:
        prediction = "Insufficient Clinical Data"
        medical_action = "Please type symptoms or upload a readable medical report to receive a diagnosis."
    else:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        medical_action = f"Based on the combined analysis, consult a specialist for evaluation of {prediction}."

    recommendations = {
        "medical_action": medical_action,
        "ecommerce_links": [f"Search local pharmacies for relevant {prediction} supplies."] if prediction != "Insufficient Clinical Data" else []
    }
    
    # Save to Database
    conn = sqlite3.connect('medical.db')
    c = conn.cursor()
    c.execute("INSERT INTO patients (name, symptoms, ocr_text, prediction, recommendations) VALUES (?, ?, ?, ?, ?)",
              (patient_name, ",".join(user_symptoms), extracted_lab_data, prediction, json.dumps(recommendations)))
    conn.commit()
    conn.close()
    
    return {
        "status": "success", 
        "data": {
            "predicted_condition": prediction,
            "report_analysis": report_analysis,
            "recommendations": recommendations,
            "ocr_extracted_text": extracted_lab_data,
            "matched_symptoms_debug": list(set(matched_triggers))
        }
    }