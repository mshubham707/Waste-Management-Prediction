
# Waste Management and Recycling Prediction System

## Overview
This project, developed for the Mini-Hackathon: Waste Management and Recycling in Indian Cities, aims to predict recycling rates based on waste management data across Indian cities. The system preprocesses raw data, trains machine learning models (Random Forest, XGBoost, Gradient Boosting), and provides predictions using a fully integrated pipeline. Built using Python, it leverages libraries like `scikit-learn`, `xgboost`, and `pandas`.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables, engineers features (e.g., distance to landfill), and scales numerical data.
- **Model Training**: Tunes and trains multiple regression models, saving the best performer.
- **Prediction**: Generates recycling rate predictions and saves results.

## Requirements
- Python 3.8+
- Required packages (install via `pip`):
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `joblib`
  - `geopy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository or download the project folder.
2. Create a virtual environment:
   ```bash
   python -m venv waste_env
   waste_env\Scripts\activate
   ```
3. Install dependencies from `requirements.txt` (create this file with the listed packages if not already done).
4. Ensure the data directory contains `data/raw/train.csv` (sample dataset provided by the hackathon).

## Project Structure
```
Waste_Management/
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py        # Data preprocessing script
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training script
│   │   └── predict.py          # Prediction script
│   └── app.py                  # Main pipeline integration script
├── data/
│   ├── raw/                    # Raw data (e.g., train.csv)
│   └── processed/              # Processed data (e.g., cleaned_data.csv)
├── models/                     # Saved model files (e.g., xgboost_tuned_model.pkl)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Usage
Run the full pipeline from the project root:
```bash
python src/app.py
```

### Output
- Processed data: Saved to `data/processed/cleaned_data.csv`.
- Trained models: Saved to `models/` (e.g., `xgboost_tuned_model.pkl`).
- Predictions: Saved to `data/predictions/predictions.csv`.

## Development
- **Notebooks**: Use `notebooks/` for exploration and prototyping.
- **Enhancements**: Add a web interface (e.g., Flask) or more feature engineering in `preprocess.py`.

## Contributing
Feel free to fork this repository, submit issues, or pull requests for improvements.

## Acknowledgements
- Thanks to the Mini-Hackathon organizers for the dataset and challenge.
- Built with help from xAI's Grok and other AI tools such as Gemini, ChatGPT used for resolving errors and refining the codes and visualizations.
```
```
