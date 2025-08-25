import pandas as pd
import numpy as np
import os

class FeatureEngineering:
    def __init__(self, dfs, interim_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/"):
        self.merged_df = dfs["merged_df"]
        self.interim_path = interim_path
        os.makedirs(self.interim_path, exist_ok=True)

    def add_budget_to_revenue_ratio(self):
        self.merged_df['budget_to_revenue_ratio'] = self.merged_df.apply(
            lambda row: row['budget'] / row['revenue'] if row['revenue'] and row['revenue'] > 0 else 0, axis=1
        )

    def add_top_genre_onehot(self, top_n=10):
        # One-hot encode the top N genres
        genre_dummies = self.merged_df['genres'].str.get_dummies(sep=', ')
        top_genres = genre_dummies.sum().sort_values(ascending=False).head(top_n).index
        for genre in top_genres:
            self.merged_df[f"genre_{genre}"] = genre_dummies[genre]

    def add_runtime_bins(self):
        bins = [0, 60, 90, 120, 180, 10000]
        labels = ['short', 'medium', 'long', 'epic', 'unknown']
        self.merged_df['runtime_bin'] = pd.cut(self.merged_df['runtime'], bins=bins, labels=labels, right=False)

    def add_log_features(self):
        # Log-transform skewed features (add 1 to avoid log(0))
        for col in ['budget', 'revenue', 'popularity', 'vote_count']:
            self.merged_df[f'log_{col}'] = np.log1p(self.merged_df[col])

    def add_interaction_features(self):
        self.merged_df['budget_x_popularity'] = self.merged_df['budget'] * self.merged_df['popularity']
        self.merged_df['budget_x_vote_count'] = self.merged_df['budget'] * self.merged_df['vote_count']

    def add_count_features(self):
        # Number of genres, keywords, cast, crew
        self.merged_df['num_genres'] = self.merged_df['genres'].fillna('').apply(lambda x: len([g for g in x.split(',') if g.strip()]))
        self.merged_df['num_keywords'] = self.merged_df['keywords'].fillna('').apply(lambda x: len([k for k in x.split(',') if k.strip()]))
        self.merged_df['num_cast'] = self.merged_df['cast'].fillna('').apply(lambda x: len([c for c in x.split(',') if c.strip()]))
        self.merged_df['num_crew'] = self.merged_df['crew'].fillna('').apply(lambda x: len([c for c in x.split(',') if c.strip()]))

    def add_text_length_features(self):
        self.merged_df['overview_length'] = self.merged_df['overview'].fillna('').apply(len)
        self.merged_df['title_length'] = self.merged_df['title'].fillna('').apply(len)

    def add_genre_mean_encoding(self):
        # Mean encoding: mean rating per genre (for top 10 genres)
        genre_ratings = {}
        for genre in self.merged_df['genres'].str.split(',').explode().str.strip().unique():
            if genre and genre != 'Unknown':
                mask = self.merged_df['genres'].str.contains(rf'\b{genre}\b', regex=True)
                genre_ratings[genre] = self.merged_df.loc[mask, 'rating'].mean()
        for genre in list(genre_ratings.keys())[:10]:
            self.merged_df[f'genre_{genre}_mean_rating'] = self.merged_df['genres'].apply(
                lambda x: genre_ratings[genre] if genre in x else np.nan
            )

    def run_all(self):
        self.add_budget_to_revenue_ratio()
        self.add_top_genre_onehot()
        self.add_runtime_bins()
        self.add_log_features()
        self.add_interaction_features()
        self.add_count_features()
        self.add_text_length_features()
        self.add_genre_mean_encoding()
        self.merged_df.to_csv(os.path.join(self.interim_path, "feature_engineered.csv"), index=False)
        print("Feature engineering complete. Data saved to interim/feature_engineered.csv")
        return self.merged_df
