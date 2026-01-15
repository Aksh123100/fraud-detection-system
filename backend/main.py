from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI(title="Credit Card Fraud Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Artifacts
MODEL_PATH = os.path.join("ml", "model.joblib")
SCALER_PATH = os.path.join("ml", "scaler.joblib")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and Scaler loaded.")
        else:
            print("Artifacts not found. Predictions will fail.")
    except Exception as e:
        print(f"Error loading model: {e}")

class TransactionRequest(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Prepare input dataframe in correct order
    # The scaler expects all 30 features (V1-V28, Time, Amount are NOT in that order in training?)
    # Wait, in training: X = df.drop('Class', axis=1).
    # Columns in df: V1...V28, Time, Amount (based on generate_data.py)
    # Let's check generate_data.py ordering.
    # df = pd.DataFrame(X, columns=cols) -> V1...V28
    # df['Time'] = ...
    # df['Amount'] = ...
    # So Columns are: V1, V2 ... V28, Time, Amount.
    # We must match this order.
    
    data_dict = transaction.dict()
    
    # Create ordered list of values
    feature_order = [f"V{i+1}" for i in range(28)] + ["Time", "Amount"]
    
    try:
        features = [[data_dict[f] for f in feature_order]]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "risk_level": "High" if probability > 0.8 else "Medium" if probability > 0.2 else "Low"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
