import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import joblib
import os

class CollaborativeFiltering:
    def __init__(self, ratings_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered.csv"):
        self.ratings_path = ratings_path
        self.df = pd.read_csv(self.ratings_path)
        self.svd_model = None
        self.knn_model = None

    def prepare_surprise_data(self):
        # Use only userId, movieId, rating for collaborative filtering
        data = self.df[['userId', 'movieId', 'rating']]
        reader = Reader(rating_scale=(data['rating'].min(), data['rating'].max()))
        dataset = Dataset.load_from_df(data, reader)
        return dataset

    def train_knn(self, k=40, sim_name='cosine', user_based=True):
        dataset = self.prepare_surprise_data()
        trainset, testset = surprise_train_test_split(dataset, test_size=0.2, random_state=42)
        sim_options = {'name': sim_name, 'user_based': user_based}
        self.knn_model = KNNBasic(k=k, sim_options=sim_options)
        self.knn_model.fit(trainset)
        preds = self.knn_model.test(testset)
        rmse = accuracy.rmse(preds, verbose=True)
        print(f"KNN ({'user' if user_based else 'item'}-based, {sim_name}) RMSE: {rmse:.4f}")
        joblib.dump(self.knn_model, os.path.join(os.path.dirname(self.ratings_path), "knn_model.joblib"))
        return self.knn_model

    def train_svd(self, n_factors=50, reg_all=0.02):
        dataset = self.prepare_surprise_data()
        trainset, testset = surprise_train_test_split(dataset, test_size=0.2, random_state=42)
        self.svd_model = SVD(n_factors=n_factors, reg_all=reg_all, random_state=42)
        self.svd_model.fit(trainset)
        preds = self.svd_model.test(testset)
        rmse = accuracy.rmse(preds, verbose=True)
        print(f"SVD MF RMSE: {rmse:.4f}")
        joblib.dump(self.svd_model, os.path.join(os.path.dirname(self.ratings_path), "svd_model.joblib"))
        return self.svd_model

    def predict(self, user_id, movie_id, model_type='svd'):
        if model_type == 'svd' and self.svd_model:
            return self.svd_model.predict(user_id, movie_id).est
        elif model_type == 'knn' and self.knn_model:
            return self.knn_model.predict(user_id, movie_id).est
        else:
            return None

    def run_all(self):
        print("Training collaborative filtering models...")
        self.train_knn()
        self.train_svd()
        print("Collaborative filtering models trained and saved.")

