import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer

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

    def add_release_date_features(self):
        # Only extract year from release_date, as month/day are not meaningful for most ML tasks
        self.merged_df['release_date'] = pd.to_datetime(self.merged_df['release_date'], errors='coerce')
        self.merged_df['release_year'] = self.merged_df['release_date'].dt.year

    def add_status_onehot(self):
        # One-hot encode 'status' (e.g., Released, Post Production, etc.)
        if 'status' in self.merged_df.columns:
            status_dummies = pd.get_dummies(self.merged_df['status'], prefix='status')
            self.merged_df = pd.concat([self.merged_df, status_dummies], axis=1)

    def add_language_onehot(self, top_n=5):
        # One-hot encode top N original languages
        if 'original_language' in self.merged_df.columns:
            lang_counts = self.merged_df['original_language'].value_counts().head(top_n).index
            for lang in lang_counts:
                self.merged_df[f'lang_{lang}'] = (self.merged_df['original_language'] == lang).astype(int)

    def add_adult_flag(self):
        # Convert 'adult' column to binary flag
        if 'adult' in self.merged_df.columns:
            self.merged_df['is_adult'] = self.merged_df['adult'].map({'True': 1, 'False': 0})

    def add_title_keyword_flags(self, keywords=['love', 'war', 'star', 'man', 'woman']):
        # Add binary flags if certain keywords appear in the title
        self.merged_df['title'] = self.merged_df['title'].fillna('').astype(str)
        for kw in keywords:
            self.merged_df[f'title_has_{kw}'] = self.merged_df['title'].str.lower().str.contains(kw).astype(int)

    def add_multi_hot_keywords(self, top_n=20):
        # Multi-hot encode top N keywords
        keywords_split = self.merged_df['keywords'].fillna('').apply(lambda x: [k.strip() for k in x.split(',') if k.strip()])
        mlb = MultiLabelBinarizer()
        top_keywords = pd.Series([k for sublist in keywords_split for k in sublist]).value_counts().head(top_n).index
        keywords_filtered = keywords_split.apply(lambda x: [k for k in x if k in top_keywords])
        keyword_dummies = pd.DataFrame(mlb.fit_transform(keywords_filtered), columns=[f'kw_{k}' for k in mlb.classes_], index=self.merged_df.index)
        self.merged_df = pd.concat([self.merged_df, keyword_dummies], axis=1)

    def add_cast_crew_features(self, top_n_cast=10, top_n_crew=10):
        # Multi-hot encode top N cast and crew
        cast_split = self.merged_df['cast'].fillna('').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
        crew_split = self.merged_df['crew'].fillna('').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
        mlb_cast = MultiLabelBinarizer()
        mlb_crew = MultiLabelBinarizer()
        top_cast = pd.Series([c for sublist in cast_split for c in sublist]).value_counts().head(top_n_cast).index
        top_crew = pd.Series([c for sublist in crew_split for c in sublist]).value_counts().head(top_n_crew).index
        cast_filtered = cast_split.apply(lambda x: [c for c in x if c in top_cast])
        crew_filtered = crew_split.apply(lambda x: [c for c in x if c in top_crew])
        cast_dummies = pd.DataFrame(mlb_cast.fit_transform(cast_filtered), columns=[f'cast_{c}' for c in mlb_cast.classes_], index=self.merged_df.index)
        crew_dummies = pd.DataFrame(mlb_crew.fit_transform(crew_filtered), columns=[f'crew_{c}' for c in mlb_crew.classes_], index=self.merged_df.index)
        self.merged_df = pd.concat([self.merged_df, cast_dummies, crew_dummies], axis=1)

    def add_company_country_features(self, top_n_company=10, top_n_country=10):
        # Multi-hot encode top N production companies and countries
        company_split = self.merged_df['production_companies'].fillna('').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
        country_split = self.merged_df['production_countries'].fillna('').apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
        mlb_company = MultiLabelBinarizer()
        mlb_country = MultiLabelBinarizer()
        top_company = pd.Series([c for sublist in company_split for c in sublist]).value_counts().head(top_n_company).index
        top_country = pd.Series([c for sublist in country_split for c in sublist]).value_counts().head(top_n_country).index
        company_filtered = company_split.apply(lambda x: [c for c in x if c in top_company])
        country_filtered = country_split.apply(lambda x: [c for c in x if c in top_country])
        company_dummies = pd.DataFrame(mlb_company.fit_transform(company_filtered), columns=[f'company_{c}' for c in mlb_company.classes_], index=self.merged_df.index)
        country_dummies = pd.DataFrame(mlb_country.fit_transform(country_filtered), columns=[f'country_{c}' for c in mlb_country.classes_], index=self.merged_df.index)
        self.merged_df = pd.concat([self.merged_df, company_dummies, country_dummies], axis=1)

    def add_temporal_features(self):
        # Remove features based on month/day/weekday/quarter, keep only year-based features
        pass  # No temporal features except year

    def add_target_encoding(self, col, target='rating', top_n=10):
        # Target mean encoding for categorical columns (e.g., genres, companies)
        values = pd.Series([v for sublist in self.merged_df[col].fillna('').apply(lambda x: [i.strip() for i in x.split(',') if i.strip()]) for v in sublist])
        top_values = values.value_counts().head(top_n).index
        for v in top_values:
            mask = self.merged_df[col].str.contains(rf'\b{v}\b', regex=True)
            mean_val = self.merged_df.loc[mask, target].mean()
            self.merged_df[f'{col}_{v}_mean_{target}'] = mask.astype(int) * mean_val

    def run_all(self):
        self.add_budget_to_revenue_ratio()
        self.add_top_genre_onehot()
        self.add_runtime_bins()
        self.add_log_features()
        self.add_interaction_features()
        self.add_count_features()
        self.add_text_length_features()
        self.add_genre_mean_encoding()
        self.add_release_date_features()
        self.add_status_onehot()
        self.add_language_onehot()
        self.add_adult_flag()
        self.add_title_keyword_flags()
        # --- Sophisticated new features ---
        self.add_multi_hot_keywords()
        self.add_cast_crew_features()
        self.add_company_country_features()
        self.add_target_encoding('genres')
        self.add_target_encoding('production_companies')
        # Save the feature-engineered DataFrame
        self.merged_df.to_csv(os.path.join(self.interim_path, "feature_engineered.csv"), index=False)
        print("Feature engineering complete. Data saved to interim/feature_engineered.csv")
        return self.merged_df
        self.add_temporal_features()
        self.add_target_encoding('genres')
        self.add_target_encoding('production_companies')
        # Save the feature-engineered DataFrame
        self.merged_df.to_csv(os.path.join(self.interim_path, "feature_engineered.csv"), index=False)
        print("Feature engineering complete. Data saved to interim/feature_engineered.csv")
        return self.merged_df
