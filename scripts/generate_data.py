import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration of simulation for Houston Standards (NSHN)
TOTAL_RECORDS = 850
UNITS = ['ICU', 'Emergency', 'Surgery', 'Pediatrics']

def create_raw_log():
    print("--- Starting generation of clinics logs ---")
    data_list = []
    start_date = datetime(2025,  1, 1)
    

    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
    
    for i in range(TOTAL_RECORDS):
        current_date = start_date + timedelta(days=np.random.randint(0, 365))
        unit = np.random.choice(UNITS)
        
        # Critical variables for the SIR (STANDARDIZED INFECTION RATIO) model
        device_days = np.random.randint(1, 30)
        observed_event = 1 if np.random.random() < 0.018 else 0
        # Variables of linical context
        patient_age = np.random.randint(18, 95)
        
        data_list.append([
            current_date, unit, patient_age, device_days, observed_event
        ])

    df = pd.DataFrame(data_list, columns=[
        'timestamp', 'care_unit', 'patient_age', 'device_exposure_days', 'clabsi_event'
    ])
    
    df.loc[np.random.choice(df.index, 10), 'patient_age'] = np.nan
    # Save
    df.to_csv('raw_data/clinical_v1_raw.csv', index=False)
    print(f"✅ filed saved with {len(df)} records in 'raw_data/clinical_v1_raw.csv'")
if __name__ == "__main__":
    create_raw_log()