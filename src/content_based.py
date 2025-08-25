import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

class ContentBasedRecommender:
    def __init__(self, feature_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered.csv"):
        self.df = pd.read_csv(feature_path)
        self.item_vectors = None
        self.tfidf = None
        self.mlb_genres = None
        self.mlb_keywords = None
        self.mlb_cast = None
        self.mlb_crew = None

    def build_item_vectors(self, n_cast=5, n_crew=5, svd_dim=None):
        # TF-IDF on overview + tagline
        text = (self.df['overview'].fillna('') + ' ' + self.df.get('tagline', '').fillna('')).values
        self.tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = self.tfidf.fit_transform(text)

        # Multi-hot genres
        genres = self.df['genres'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
        self.mlb_genres = MultiLabelBinarizer()
        genres_matrix = self.mlb_genres.fit_transform(genres)

        # Multi-hot keywords
        keywords = self.df['keywords'].fillna('').apply(lambda x: [k.strip() for k in x.split(',') if k.strip()])
        self.mlb_keywords = MultiLabelBinarizer()
        keywords_matrix = self.mlb_keywords.fit_transform(keywords)

        # Multi-hot top-k cast
        cast = self.df['cast'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')[:n_cast] if c.strip()])
        self.mlb_cast = MultiLabelBinarizer()
        cast_matrix = self.mlb_cast.fit_transform(cast)

        # Multi-hot top-k crew
        crew = self.df['crew'].fillna('').apply(lambda x: [c.strip() for c in x.split(',')[:n_crew] if c.strip()])
        self.mlb_crew = MultiLabelBinarizer()
        crew_matrix = self.mlb_crew.fit_transform(crew)

        # Concatenate all features
        item_matrix = np.hstack([tfidf_matrix.toarray(), genres_matrix, keywords_matrix, cast_matrix, crew_matrix])

        # Optional: Dimensionality reduction
        if svd_dim is not None and item_matrix.shape[1] > svd_dim:
            svd = TruncatedSVD(n_components=svd_dim, random_state=42)
            item_matrix = svd.fit_transform(item_matrix)

        self.item_vectors = item_matrix
        print(f"Item vectors shape: {item_matrix.shape}")
        return item_matrix

    def build_user_profile(self, user_id, rating_col='rating'):
        # Aggregate item vectors for items rated by user, weighted by rating - user mean
        user_data = self.df[self.df['userId'] == user_id]
        if user_data.empty:
            return None
        mean_rating = user_data[rating_col].mean()
        indices = user_data.index
        weights = user_data[rating_col] - mean_rating
        profile = np.average(self.item_vectors[indices], axis=0, weights=weights)
        return profile

    def recommend(self, user_id, top_n=10):
        profile = self.build_user_profile(user_id)
        if profile is None:
            # Cold-start: fallback to popularity
            top_items = self.df.sort_values('popularity', ascending=False).head(top_n)
            return top_items[['movieId', 'title', 'popularity']]
        # Compute cosine similarity
        sims = self.item_vectors @ profile / (np.linalg.norm(self.item_vectors, axis=1) * np.linalg.norm(profile) + 1e-8)
        top_indices = np.argsort(sims)[-top_n:][::-1]
        return self.df.iloc[top_indices][['movieId', 'title', 'genres', 'cast', 'crew', 'overview']]

    def explain(self, user_id, item_id):
        # Provide a simple explanation based on shared genres/cast/crew
        user_data = self.df[self.df['userId'] == user_id]
        item_row = self.df[self.df['movieId'] == item_id].iloc[0]
        user_genres = set(','.join(user_data['genres'].fillna('')).split(','))
        item_genres = set(item_row['genres'].split(','))
        shared_genres = user_genres & item_genres
        user_cast = set(','.join(user_data['cast'].fillna('')).split(','))
        item_cast = set(item_row['cast'].split(','))
        shared_cast = user_cast & item_cast
        explanation = []
        if shared_genres:
            explanation.append(f"Shared genres: {', '.join([g for g in shared_genres if g])}")
        if shared_cast:
            explanation.append(f"Shared cast: {', '.join([c for c in shared_cast if c])}")
        return " | ".join(explanation) if explanation else "Recommended based on your preferences."

