import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_synthetic_data(n_samples=10000, save_path="data/creditcard_synthetic.csv"):
    print(f"Generating {n_samples} synthetic transactions...")
    
    # Generate synthetic features (V1-V28) using make_classification to mimic fraud patterns
    # weights=[0.99, 0.01] mimics the high imbalance of real credit card data
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=28, 
        n_informative=24, 
        n_redundant=4, 
        n_classes=2, 
        weights=[0.99, 0.01], 
        random_state=42
    )
    
    # Create DataFrame
    cols = [f"V{i+1}" for i in range(28)]
    df = pd.DataFrame(X, columns=cols)
    
    # Add 'Time' feature (simulation: 0 to 48 hours in seconds)
    df['Time'] = np.random.randint(0, 172800, size=n_samples)
    
    # Add 'Amount' feature (log-normal distribution to mimic transaction amounts)
    # Most transactions are small, some are very large
    df['Amount'] = np.random.lognormal(mean=2, sigma=1.0, size=n_samples)
    
    # Add 'Class' target (0 = Legitimate, 1 = Fraud)
    df['Class'] = y
    
    # Sort by Time
    df = df.sort_values(by='Time').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    print(df['Class'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data()
