# baseline_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Data
df = pd.read_csv("./data/processed_telco_data.csv")

# Feature Selection
class FeatureSelector:
    def __init__(self, df, target_col='Churn Label'):
        self.df = df.copy()
        self.target_col = target_col
        self.le = LabelEncoder()
        
        # Ensure only numerical columns are used
        self.numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
    
    def prepare_target(self):
        """Prepare target variable for selection methods."""
        return self.le.fit_transform(self.df[self.target_col])
    def boruta_selection(self):
        """Select features using Boruta algorithm."""
        y = self.prepare_target()
        X = self.df[self.numerical_cols]
        
        # Initialize Random Forest classifier
        rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
        
        # Initialize Boruta
        boruta = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
        
        # Fit Boruta
        boruta.fit(X.values, y)
        
        # Get selected features
        selected_features = X.columns[boruta.support_].tolist()
        feature_ranks = pd.DataFrame({
            'Feature': X.columns,
            'Boruta_Ranking': boruta.ranking_
        }).sort_values('Boruta_Ranking')
        
        logger.info(f"Selected {len(selected_features)} features using Boruta")
        return selected_features, feature_ranks

if __name__ == "__main__":
    df = df 
    selector = FeatureSelector(df)
    selected_boruta, boruta_ranks = selector.boruta_selection()
    
    # Save features
    joblib.dump(selected_boruta, "baseline_boruta_features.pkl")

    print("Top Features from Boruta:", selected_boruta)

# Prepare Data
selected_features = joblib.load('baseline_boruta_features.pkl')
X = df[selected_features]
y = df["Churn Label"].apply(lambda x: 1 if x == "Yes" else 0)  # Convert target to binary

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "baseline_scaler.pkl")

# Define Model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "baseline_churn_model.pkl")

# Evaluate Model
y_pred = best_model.predict(X_test_scaled)
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
