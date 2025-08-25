import pandas as pd
import numpy as np
import joblib
from src.content_based import ContentBasedRecommender

class HybridRecommender:
    def __init__(self, feature_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered.csv",
                 alpha=0.5):
        self.feature_path = feature_path
        self.df = pd.read_csv(self.feature_path)
        self.cb = ContentBasedRecommender(feature_path)
        self.cb.build_item_vectors()
        self.svd_model = joblib.load("D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/svd_model.joblib")
        self.alpha = alpha

    def hybrid_score(self, user_id, movie_id):
        # Content-based score: cosine similarity between user profile and item
        profile = self.cb.build_user_profile(user_id)
        item_idx = self.df[self.df['movieId'] == movie_id].index
        if len(item_idx) == 0 or profile is None:
            cb_score = 0
        else:
            item_vec = self.cb.item_vectors[item_idx[0]]
            cb_score = np.dot(profile, item_vec) / (np.linalg.norm(profile) * np.linalg.norm(item_vec) + 1e-8)
        # Collaborative filtering score: SVD predicted rating (normalized)
        try:
            cf_score = self.svd_model.predict(user_id, movie_id).est
        except Exception:
            cf_score = 0
        # Normalize scores
        cb_score_norm = (cb_score + 1) / 2  # cosine similarity [-1,1] to [0,1]
        cf_score_norm = (cf_score - 0) / 5  # assuming ratings 0-5
        return self.alpha * cf_score_norm + (1 - self.alpha) * cb_score_norm

    def recommend(self, user_id, top_n=10):
        movie_ids = self.df['movieId'].unique()
        scores = []
        for mid in movie_ids:
            score = self.hybrid_score(user_id, mid)
            scores.append((mid, score))
        top_movies = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        movie_df = self.df.set_index('movieId')
        results = []
        for mid, score in top_movies:
            if mid in movie_df.index:
                row = movie_df.loc[mid]
                results.append({'movieId': mid, 'title': row['title'], 'score': score})
        return pd.DataFrame(results)

    def run_all(self, user_id=1, top_n=10):
        print(f"Hybrid recommendations for user {user_id}:")
        recs = self.recommend(user_id, top_n)
        print(recs)
        return recs

