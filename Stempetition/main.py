from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import numpy as np
import pickle

app = FastAPI()
stroke_model = pickle.load(open("E:\Python\Stempetition\StrokePrediction.pkl", "rb"))


class StrokeFeature(BaseModel):
    bmi: float
    gender: str
    hypertension: bool
    heart_disease: bool
    ever_married: bool
    work_type: str
    residence_type: str
    avg_glucose_level: float
    smoking_status: str
    age: int
    children: int

job_type_decode = {
    "self-employed": [0,0,0,1],
    "govt-job": [1,0,0,0],
    "never-work": [0,1,0,0],
    "private": [0,0,0,1]
}

smoking_status_decode = {
    "unknown": [1,0,0,0],
    "formerly_smoked": [0,1,0,0],
    "never_smoked": [0,0,1,0],
    "smokes": [0,0,1,0]
}

@app.post("/predict_stroke")
async def predict_stroke(stroke_feature: StrokeFeature):
    feature = [
        1 if stroke_feature.gender == 'male' else 0,
        stroke_feature.age,
        1 if stroke_feature.hypertension else 0,
        1 if stroke_feature.heart_disease else 0,
        1 if stroke_feature.ever_married else 0,
        1 if stroke_feature.residence_type == 'urban' else 0,
        stroke_feature.avg_glucose_level,
        stroke_feature.bmi,
    ]
    feature.extend(job_type_decode[stroke_feature.work_type])
    feature.append(stroke_feature.children)
    feature.extend(smoking_status_decode[stroke_feature.smoking_status])
    
    feature = np.array(feature).reshape(1, -1)

    prediction = np.argmax(stroke_model.predict(feature),axis=1)
    if prediction[0] == 1:
        return {"prediction": "stroke"}
    else:
        return {"prediction": "no stroke"}