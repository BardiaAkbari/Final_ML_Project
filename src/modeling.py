import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class Modeling:
    def __init__(self, feature_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered_with_ratings.csv"):
        self.feature_path = feature_path
        self.df = pd.read_csv(self.feature_path)
        self.model = None

    def prepare_data(self, target_column='rating', test_size=0.2, random_state=42):
        # Drop non-numeric and identifier columns
        drop_cols = [
            'title', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'crew',
            'production_companies', 'production_countries', 'spoken_languages', 'status',
            'release_date', 'runtime_bin', 'adult', 'timestamp', 'imdbId', 'id', 'tmdbId', 'userId', 'movieId'
        ]
        X = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns] + [target_column], errors='ignore')
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        y = self.df[target_column]
        # Remove rows with missing target
        mask = y.notnull()
        X, y = X[mask], y[mask]
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Impute any remaining NaNs (after scaling) with 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test, X.columns

    def train_model(self, X_train, y_train):
        # Use a more robust model and tune hyperparameters
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, feature_names):
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R2: {r2:.4f}")
        # Print top 10 feature importances
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        print("Top 10 features:")
        for i in top_idx:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
        return mse, r2

    def run_all(self):
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data()
        self.train_model(X_train, y_train)
        self.evaluate(X_test, y_test, feature_names)
        # Save the model
        model_path = os.path.join(os.path.dirname(self.feature_path), "gb_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
