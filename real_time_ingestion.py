import os
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

def stream_data(file_path, batch_size=20):
    """
    Generator function to simulate streaming data from a CSV file.
    Yields batches of rows (as a DataFrame) every few seconds.
    """
    # Read the CSV file in chunks
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        yield chunk
        # Simulate delay between batches (e.g., 1 second)
        time.sleep(1)

def main():
    base_dir = os.getcwd()
    preprocessed_dir = os.path.join(base_dir, "preprocessed_data")
    test_file = os.path.join(preprocessed_dir, "test_preprocessed.csv")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    # Load the supervised model (Random Forest)
    supervised_model_file = os.path.join(base_dir, "rf_model.joblib")
    if not os.path.exists(supervised_model_file):
        print(f"Supervised model file not found: {supervised_model_file}")
        return
    rf_model = joblib.load(supervised_model_file)
    print("Loaded supervised model from:", supervised_model_file)
    
    # Stream data from test file
    print("Starting data stream simulation...")
    for batch in stream_data(test_file, batch_size=20):
        # Prepare the features: assume test data has same columns and no label column needed
        # (If the CSV includes a label column, drop it)
        if 'label' in batch.columns:
            X_batch = batch.drop("label", axis=1)
        else:
            X_batch = batch
        
        # Supervised prediction (known threat classification)
        supervised_preds = rf_model.predict(X_batch)
        
        # Display predictions for this batch
        print("\nBatch Predictions:")
        print("Supervised model predictions:")
        print(supervised_preds)
        
        # Optional: store these results for further analysis

if __name__ == "__main__":
    main()
