import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

class RecommenderModels:
    def __init__(self, merged_df_with_tfidf, unique_movies_reduced, ratings_df):
        self.merged_df_with_tfidf = merged_df_with_tfidf
        self.unique_movies_reduced = unique_movies_reduced
        self.ratings_df = ratings_df
        self.popular_movies_unique = None
        self.user_profiles = None
        self.knn_user_based = None
        self.svd_mf = None
        self.svd_mf_tuned = None
        self.best_alpha = None

    # ---------- Popularity Baseline ----------
    def fit_popularity(self):
        C = self.unique_movies_reduced['vote_average'].mean()
        m = self.unique_movies_reduced['vote_count'].quantile(0.90)
        qualified = self.unique_movies_reduced[self.unique_movies_reduced['vote_count'] >= m].copy()
        def weighted_rating(x):
            v, R = x['vote_count'], x['vote_average']
            return (v / (v + m) * R) + (m / (v + m) * C)
        qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
        popular = qualified.sort_values('weighted_rating', ascending=False)
        self.popular_movies_unique = popular.groupby('movieId').first().reset_index()

    # ---------- Content-Based ----------
    def fit_content_based(self):
        movie_id_to_index = pd.Series(self.unique_movies_reduced.index, index=self.unique_movies_reduced['movieId']).to_dict()
        svd_features = self.unique_movies_reduced.filter(like='svd_')
        self.user_profiles = {}
        for user_id in self.unique_movies_reduced['userId'].unique():
            user_ratings = self.unique_movies_reduced[self.unique_movies_reduced['userId'] == user_id][['movieId', 'rating']]
            profile = np.zeros(svd_features.shape[1])
            total_weight = 0
            for _, row in user_ratings.iterrows():
                idx = movie_id_to_index.get(int(row['movieId']))
                if idx is not None:
                    profile += svd_features.loc[idx].values * row['rating']
                    total_weight += row['rating']
            if total_weight > 0:
                profile /= total_weight
            self.user_profiles[user_id] = profile

    def get_content_based_recommendations(self, user_id, top_n=10):
        if self.user_profiles is None:
            raise ValueError("Call fit_content_based() first.")
        if user_id not in self.user_profiles or np.all(self.user_profiles[user_id] == 0):
            if self.popular_movies_unique is not None:
                return self.popular_movies_unique[['title', 'vote_count', 'vote_average', 'weighted_rating']].head(top_n)
            return pd.DataFrame()
        user_profile = self.user_profiles[user_id]
        svd_features = self.unique_movies_reduced.filter(like='svd_')
        sim_scores = cosine_similarity(user_profile.reshape(1, -1), svd_features)[0]
        rated_ids = self.merged_df_with_tfidf[self.merged_df_with_tfidf['userId'] == user_id]['movieId'].tolist()
        indices = [i for i, row in self.unique_movies_reduced.iterrows() if row['movieId'] not in rated_ids]
        top_indices = np.argsort(sim_scores[indices])[::-1][:top_n]
        recs = self.unique_movies_reduced.iloc[[indices[i] for i in top_indices]][['title', 'vote_average', 'vote_count']]
        return recs.reset_index(drop=True)

    def get_content_based_score(self, user_id, movie_id):
        if self.user_profiles is None:
            raise ValueError("Call fit_content_based() first.")
        if user_id not in self.user_profiles or np.all(self.user_profiles[user_id] == 0):
            return 0.0
        user_profile = self.user_profiles[user_id]
        idx = self.unique_movies_reduced[self.unique_movies_reduced['movieId'] == movie_id].index
        if idx.empty:
            return 0.0
        movie_features = self.unique_movies_reduced.loc[idx].filter(like='svd_').values
        return cosine_similarity(user_profile.reshape(1, -1), movie_features.reshape(1, -1))[0][0]

    # ---------- Collaborative Filtering ----------
    def fit_cf(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], reader)
        self.data = data
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)
        self.knn_user_based = KNNBasic(sim_options={'user_based': True, 'similarity': 'cosine'}, k=40)
        self.knn_user_based.fit(self.trainset)
        self.svd_mf = SVD(random_state=42)
        self.svd_mf.fit(self.trainset)

    def evaluate_cf(self):
        preds_knn = self.knn_user_based.test(self.testset)
        preds_svd = self.svd_mf.test(self.testset)
        rmse_knn = accuracy.rmse(preds_knn)
        rmse_svd = accuracy.rmse(preds_svd)
        return rmse_knn, rmse_svd

    # ---------- Hybrid Model ----------
    def hybrid_prediction(self, user_id, movie_id, alpha):
        cb_score = self.get_content_based_score(user_id, movie_id)
        try:
            cf1_pred = self.knn_user_based.predict(str(user_id), str(movie_id)).est
        except Exception:
            cf1_pred = 0
        try:
            cf2_pred = self.svd_mf.predict(str(user_id), str(movie_id)).est
        except Exception:
            cf2_pred = 0
        cf_score = (cf1_pred + cf2_pred) / 2.0
        return alpha * cf_score + (1 - alpha) * cb_score

    def tune_hybrid_alpha(self, alphas=None):
        if alphas is None:
            alphas = np.arange(0, 1.01, 0.5)
        testset_df = pd.DataFrame(self.testset, columns=['userId', 'movieId', 'rating'])
        # Recreate user profiles from trainset
        train_ratings_df = pd.DataFrame(self.trainset.all_ratings(), columns=['uid', 'iid', 'rating'])
        train_ratings_df['userId'] = train_ratings_df['uid'].apply(lambda x: self.trainset.to_raw_uid(x))
        train_ratings_df['movieId'] = train_ratings_df['iid'].apply(lambda x: self.trainset.to_raw_iid(x))
        train_ratings_df = train_ratings_df[['userId', 'movieId', 'rating']]
        self.fit_content_based()  # Ensure user_profiles is up to date
        rmse_scores = {}
        for alpha in alphas:
            preds, actuals = [], []
            for _, row in testset_df.iterrows():
                pred = self.hybrid_prediction(int(row['userId']), int(row['movieId']), alpha)
                preds.append(pred)
                actuals.append(row['rating'])
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            rmse_scores[alpha] = rmse
        self.best_alpha = min(rmse_scores, key=rmse_scores.get)
        return rmse_scores, self.best_alpha

    def fit_svd_gridsearch(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_factors': [50, 100, 150],
                'lr_all': [0.002, 0.005, 0.01],
                'reg_all': [0.02, 0.05, 0.1]
            }
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
        gs.fit(self.data)
        self.svd_mf_tuned = SVD(**gs.best_params['rmse'])
        self.svd_mf_tuned.fit(self.data.build_full_trainset())
        return gs.best_score['rmse'], gs.best_params['rmse']

    def evaluate_hybrid(self):
        testset_df = pd.DataFrame(self.testset, columns=['userId', 'movieId', 'rating'])
        preds, actuals = [], []
        for _, row in testset_df.iterrows():
            pred = self.hybrid_prediction(int(row['userId']), int(row['movieId']), self.best_alpha)
            preds.append(pred)
            actuals.append(row['rating'])
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        return rmse

# Example usage:
# models = RecommenderModels(merged_df_with_tfidf, unique_movies_reduced, ratings_df)
# models.fit_popularity()
# models.fit_content_based()
# models.fit_cf()
# print(models.evaluate_cf())
# rmse_scores, best_alpha = models.tune_hybrid_alpha()
# print("Best alpha:", best_alpha)
# print("Hybrid RMSE:", models.evaluate_hybrid())