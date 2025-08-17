import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions from your scripts
from data.preprocess import preprocess_data
from models.train import train_models  
from models.predict import predict_recycling_rate

#Main Function
def run_pipeline():
    print("Starting full pipeline integration...")
    
    # Step 1: Preprocessing
    print("\n--- Step 1: Preprocessing ---")
    preprocess_data(data='data/raw/train.csv', is_file=True, output_path='data/processed/cleaned_data.csv')
    
    # Step 2: Training
    print("\n--- Step 2: Training ---")
    train_models()  # This will run the training logic from train.py
    
    # Step 3: Prediction
    print("\n--- Step 3: Prediction ---")
    predict_recycling_rate(data='data/processed/cleaned_data.csv', is_file=True, processing_needed=False)  # Uses the trained model
    
    print("\nPipeline completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    run_pipeline()