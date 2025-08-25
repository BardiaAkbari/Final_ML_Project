import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class Modeling:
    def __init__(self, feature_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered.csv"):
        self.feature_path = feature_path
        self.df = pd.read_csv(self.feature_path)
        self.model = None

    def prepare_data(self, target_column='rating', test_size=0.2, random_state=42):
        # Drop non-numeric and identifier columns
        drop_cols = [
            'title', 'original_title', 'overview', 'genres', 'keywords', 'cast', 'crew',
            'production_companies', 'production_countries', 'spoken_languages', 'status',
            'release_date', 'runtime_bin', 'adult', 'timestamp'
        ]
        X = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns] + [target_column, 'userId', 'movieId'], errors='ignore')
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R2: {r2:.4f}")
        return mse, r2

    def run_all(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.train_model(X_train, y_train)
        self.evaluate(X_test, y_test)
        # Save the model
        model_path = os.path.join(os.path.dirname(self.feature_path), "rf_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
