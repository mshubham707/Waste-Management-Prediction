import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocess import preprocess_data

# Load the model
print("Loading Model..")
def load_model(model_path='models/xgboost_tuned_model.pkl'):
    print("Model Loaded Successfully.")
    return joblib.load(model_path)

# Predict recycling rate
def predict_recycling_rate(data, is_file=False, processing_needed=True, model_path='models/xgboost_tuned_model.pkl'):
    """
    Predict the recycling rate using the trained model.
    Args:
        data (pd.DataFrame or str): The input data for prediction, either as a DataFrame or a file path.
        is_file (bool): Flag indicating if the input data is a file path.
        model_path (str): The file path to the trained model.
    """

    print("Predicting Recycling Rate...")

    
    model = load_model(model_path)
    if is_file:
        df = pd.read_csv(data)
    else:
        df = pd.DataFrame(data)

    if processing_needed:
        df = preprocess_data(data=df, is_file=False, output_path='data/processed/cleaned_data.csv')


    df.drop("Recycling_Rate", axis=1, inplace=True)
    prediction = model.predict(df)
    print("Prediction completed.")
    predictions = pd.Series(prediction, name='Recycling_Rate')
    saved_path = 'data/predictions/predictions.csv'
    os.makedirs('data/predictions', exist_ok=True)  # Create directory if needed
    predictions.to_csv(saved_path, index=False)
    print(f"Predictions saved to {saved_path}")
    return predictions

# Example usage (for testing)
if __name__ == "__main__":
    print("Starting prediction...")
    predict_recycling_rate(data='data/raw/train.csv',  is_file=True,processing_needed=True)