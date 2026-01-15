import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_model(data_path="data/creditcard_synthetic.csv", model_dir="ml"):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data (80% train, 20% test)
    # stratify=y is crucial for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
    # We should scale 'Amount' and 'Time'. V1-V28 are usually already scaled (PCA features), 
    # but for synthetic data generated with make_classification, they are centered.
    # It's good practice to scale everything for many models, though Random Forest is robust to unscaled data.
    # However, since we might want to swap models later (e.g. Logistic Regression), let's scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model
    print("Training Random Forest Classifier (this may take a moment)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    print(f"Model and scaler saved to {model_dir}")

if __name__ == "__main__":
    train_model()
