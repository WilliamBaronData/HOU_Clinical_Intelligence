import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_hospital_model():
    # --- SMART ROUTE CONFIGURATION ---
    # This gets the path to the folder where this script is located (scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # This moves up one level to the project's root folder (HOU_Clinical_Analytics_V1)
    project_root = os.path.dirname(script_dir)

    # We defined the exact routes
    data_path = os.path.join(project_root, 'raw_data', 'clinical_v1_raw.csv')
    output_dir = os.path.join(project_root, 'data')
    output_path = os.path.join(output_dir, 'clinical_predictions_forbi.csv')

    # 1. Load the data we generated earlier.
    if not os.path.exists(data_path):
        print(f"❌ error: The file was not found in {data_path}")
        print("Make sure you have run the following command first: python3 scripts/generate_data.py")
        return

    df = pd.read_csv(data_path)

    # 2. Cleaning and Preparation
    # We save a copy for the final Power BI file
    df_results =df.copy()

    # We converted the hospital categories into numbers.
    df['care_unit_code'] = df['care_unit'].astype('category').cat.codes

    # We filled in the missing ages with the average.
    df['patient_age'] = df['patient_age'].fillna(df['patient_age'].mean())

    # 3. Define what we want to predict.
    # Variables X (predictors) and y (target)
    X = df[['care_unit_code', 'patient_age', 'device_exposure_days']]
    y = df['clabsi_event']

    # 4. Divide the data for testing: 80% for training, 20% for the final test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train the model (Random Forest is the standard due to its interpretability)
    print('---Training the clinical AI Model ---')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Assess
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    # 7. Generate predictions for Power BI
    # We calculated the probability of risk (0.0 to 1.0) for all records.
    df_results['risk_probability'] = model.predict_proba(X)[: , 1]

    # We create the output folder if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We saved the final file enhanced with AI
    df_results.to_csv(output_path, index=False)

    print(f"📊 Successful export for Power BI at: {output_path}")

if __name__ == "__main__":
    train_hospital_model()