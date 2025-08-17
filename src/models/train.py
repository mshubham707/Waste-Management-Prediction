import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import  r2_score, root_mean_squared_error
import joblib

# Load and prepare data
def load_data():
    print("Loading data...")
    df = pd.read_csv('data/processed/cleaned_data.csv')
    X = df.drop(columns=['Recycling_Rate'])  
    y = df['Recycling_Rate']
    print("Data shape:", df.shape)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids
param_grids = {
    'Random Forest': {
        'estimator': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
    },
    'XGBoost': {
        'estimator': XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'min_child_weight': [1, 5]
        }
    },
    'Gradient Boosting': {
        'estimator': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    }
}

# Train and tune models
def train_models():
    print("Training and tuning models...")
    X_train, X_test, y_train, y_test = load_data()
    tuned_models = pd.DataFrame(columns=['Model','Parameters', 'RMSE', 'R2'])
    best_score = -float('inf')
    best_grid_search = None

    for model_name, config in param_grids.items():
        print(f"\n🔍 Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=config['estimator'],
            param_grid=config['params'],
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        y_pred = grid_search.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        tuned_models.loc[len(tuned_models)] = {'Model': model_name, 'Parameters': grid_search.best_params_, 'RMSE': rmse, 'R2': r2}
        print(f"✅ Best Parameters: {grid_search.best_params_}")
        print(f"📊 RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_grid_search = grid_search

    tuned_models = tuned_models.sort_values(by='R2', ascending=False).reset_index(drop=True)
    print("\nTuned Models Results:")
    print(tuned_models)
    
    # Save the best model
    best_model_name = tuned_models.iloc[0]['Model']
    best_model = best_grid_search.best_estimator_
    joblib.dump(best_model, f'models/{best_model_name.lower().replace(" ", "_")}_tuned_model.pkl')
    print(f"Best model ({best_model_name}) saved to models/{best_model_name.lower().replace(' ', '_')}_tuned_model.pkl")
    print("Training complete.")
    return tuned_models, best_grid_search

# Main execution (for standalone testing)
if __name__ == "__main__":
    train_models()