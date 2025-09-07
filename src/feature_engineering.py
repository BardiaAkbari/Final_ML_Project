import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD

class FeatureEngineering:
    def __init__(self, dfs, interim_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/"):
        self.merged_df = dfs["merged_df"]
        self.ratings_df = dfs["ratings_df"]
        self.interim_path = interim_path
        os.makedirs(self.interim_path, exist_ok=True)
    
<<<<<<< HEAD
    def ordering(self):
        self.merged_df = self.merged_df.drop(columns=['id', 'tmdbId', 'imdbId', 'imdb_id', 'original_title', 'video'])
        desired_column_order = [
            'movieId',
            'title',
            'release_date',
            'runtime',
            'status',
            'adult',
            'budget',
            'revenue',
            'popularity',
            'vote_average',
            'vote_count',
            'overview',
            'genres',
            'keywords',
            'cast',
            'crew',
            'production_companies',
            'production_countries',
            'original_language',
            'userId',
            'rating',
        ]

        self.merged_df = self.merged_df.reindex(columns=desired_column_order)
    
=======
>>>>>>> 848b2a5 (almost finilaize)
    def outliers(self):
        self.merged_df['budget'] = pd.to_numeric(self.merged_df['budget'], errors='coerce').fillna(0)
        self.merged_df['revenue'] = pd.to_numeric(self.merged_df['revenue'], errors='coerce').fillna(0)
        self.merged_df = self.merged_df[self.merged_df['runtime'] > 0]
        self.merged_df = self.merged_df[self.merged_df['budget'] >= 0]
        self.merged_df = self.merged_df[self.merged_df['revenue'] >= 0]
        
        for col in ['budget', 'revenue']:
            upper = self.merged_df[col].quantile(0.995)
            self.merged_df = self.merged_df[self.merged_df[col] <= upper]
    
    def add_budget_to_revenue_ratio(self):
        self.merged_df['budget'] = pd.to_numeric(self.merged_df['budget'], errors='coerce').fillna(0)
        self.merged_df['revenue'] = pd.to_numeric(self.merged_df['revenue'], errors='coerce').fillna(0)
        self.merged_df['budget_to_revenue_ratio'] = self.merged_df.apply(
            lambda row: row['budget'] / row['revenue'] if row['revenue'] > 0 else 0, axis=1
        )

    def add_top_genre_onehot(self, top_n=5):
        genre_dummies = self.merged_df['genres'].str.get_dummies(sep=', ')
        top_genres = genre_dummies.sum().sort_values(ascending=False).head(top_n).index
        for genre in top_genres:
            self.merged_df[f"genre_{genre}"] = genre_dummies[genre]

    def add_log_features(self):
        for col in ['budget', 'revenue', 'popularity', 'vote_count']:
            self.merged_df[f'log_{col}'] = np.log1p(self.merged_df[col])

    def add_interaction_features(self):
        self.merged_df['budget_x_popularity'] = self.merged_df['budget'] * self.merged_df['popularity']
        self.merged_df['budget_x_vote_count'] = self.merged_df['budget'] * self.merged_df['vote_count']

    def add_count_features(self):
        self.merged_df['num_genres'] = self.merged_df['genres'].fillna('').apply(lambda x: len([g for g in x.split(',') if g.strip()]))
        self.merged_df['num_keywords'] = self.merged_df['keywords'].fillna('').apply(lambda x: len([k for k in x.split(',') if k.strip()]))
        self.merged_df['num_cast'] = self.merged_df['cast'].fillna('').apply(lambda x: len([c for c in x.split(',') if c.strip()]))
        self.merged_df['num_crew'] = self.merged_df['crew'].fillna('').apply(lambda x: len([c for c in x.split(',') if c.strip()]))

<<<<<<< HEAD
    def add_text_length_features(self):
        self.merged_df['overview_length'] = self.merged_df['overview'].fillna('').apply(len)
        self.merged_df['title_length'] = self.merged_df['title'].fillna('').apply(len)

    def add_genre_mean_encoding(self):
        genre_ratings = {}
        for genre in self.merged_df['genres'].str.split(',').explode().str.strip().unique():
            if genre and genre != 'Unknown':
                mask = self.merged_df['genres'].str.contains(rf'\b{genre}\b', regex=True)
                genre_ratings[genre] = self.merged_df.loc[mask, 'vote_average'].mean()
        for genre in list(genre_ratings.keys())[:10]:
            self.merged_df[f'genre_{genre}_mean_vote'] = self.merged_df['genres'].apply(
                lambda x: genre_ratings[genre] if genre in x else np.nan
            )

=======
>>>>>>> 848b2a5 (almost finilaize)
    def add_release_date_features(self):
        self.merged_df['release_date'] = pd.to_datetime(self.merged_df['release_date'], errors='coerce')
        self.merged_df['release_year'] = self.merged_df['release_date'].dt.year
        self.merged_df.drop(columns=['release_date'], inplace=True)


    def add_adult_flag(self):
        if 'adult' in self.merged_df.columns:
            self.merged_df['is_adult'] = self.merged_df['adult'].map({'True': 1, 'False': 0})
        self.merged_df.drop(columns=['adult'], inplace=True)

    def add_multi_hot_keywords(self, top_n=20):
        keywords_split = self.merged_df['keywords'].fillna('').apply(lambda x: [k.strip() for k in x.split(',') if k.strip()])
        mlb = MultiLabelBinarizer()
        top_keywords = pd.Series([k for sublist in keywords_split for k in sublist]).value_counts().head(top_n).index
        keywords_filtered = keywords_split.apply(lambda x: [k for k in x if k in top_keywords])
        keyword_dummies = pd.DataFrame(mlb.fit_transform(keywords_filtered), columns=[f'kw_{k}' for k in mlb.classes_], index=self.merged_df.index)
        self.merged_df = pd.concat([self.merged_df, keyword_dummies], axis=1)

    def add_cast_crew_features(self, top_n_cast=5, top_n_crew=5):
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

    def add_company_country_features(self, top_n_company=5, top_n_country=5):
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

<<<<<<< HEAD
=======
    def status_onehot(self):
        self.merged_df = pd.get_dummies(self.merged_df, columns=['status'], drop_first=False)

>>>>>>> 848b2a5 (almost finilaize)
    def add_target_encoding(self, col, target='vote_average', top_n=10):
        values = pd.Series([v for sublist in self.merged_df[col].fillna('').apply(lambda x: [i.strip() for i in x.split(',') if i.strip()]) for v in sublist])
        top_values = values.value_counts().head(top_n).index
        for v in top_values:
            mask = self.merged_df[col].str.contains(rf'\b{v}\b', regex=True)
            mean_val = self.merged_df.loc[mask, target].mean()
            self.merged_df[f'{col}_{v}_mean_{target}'] = mask.astype(int) * mean_val
    
    def coding(self):
        self.add_target_encoding(col='genres')
        self.add_target_encoding(col='production_companies')
<<<<<<< HEAD
    
    def Tfidf(self):
        tfidf_overview_vectorizer = TfidfVectorizer(max_features=2100, stop_words='english')
=======
        
    
    def Tfidf(self):
        tfidf_overview_vectorizer = TfidfVectorizer(max_features=2600, stop_words='english')
>>>>>>> 848b2a5 (almost finilaize)
        tfidf_overview_matrix = tfidf_overview_vectorizer.fit_transform(self.merged_df['overview'].fillna(''))
        self.tfidf_overview_df = pd.DataFrame(tfidf_overview_matrix.toarray(), columns=[f'overview_tfidf_{col}' for col in tfidf_overview_vectorizer.get_feature_names_out()], index=self.merged_df.index)
        
    def merging_Tfidf(self):
        # Combine the original dataframe with the TF-IDF features
        self.merged_df_with_tfidf = pd.concat([self.merged_df, self.tfidf_overview_df], axis=1)
        
    def presvd(self):
        columns_for_svd = self.merged_df_with_tfidf.select_dtypes(include=np.number).columns.tolist()
        columns_for_svd = [col for col in columns_for_svd if col not in ['rating', 'movieId', 'userId', 'timestamp', 'release_year']] # Exclude non-feature columns and year

        for col in columns_for_svd:
            if self.merged_df_with_tfidf[col].isnull().any():
                median_val = self.merged_df_with_tfidf[col].median()
                self.merged_df_with_tfidf[col] = self.merged_df_with_tfidf[col].fillna(median_val)
        if 'production_companies_Warner Bros._mean_vote_average' in self.merged_df_with_tfidf.columns:
            self.merged_df_with_tfidf['production_companies_Warner Bros._mean_vote_average'] = self.merged_df_with_tfidf['production_companies_Warner Bros._mean_vote_average'].fillna(0)
        
    
    def svd(self):
        unique_movies_df = self.merged_df_with_tfidf.groupby('movieId').first().reset_index()
        columns_for_svd_unique = unique_movies_df.select_dtypes(include=np.number).columns.tolist()
        columns_for_svd_unique = [col for col in columns_for_svd_unique if col not in ['rating', 'movieId', 'userId', 'timestamp', 'release_year', 'vote_average', 'vote_count']]

<<<<<<< HEAD
=======
        # Fill NaNs with median for all SVD columns
>>>>>>> 848b2a5 (almost finilaize)
        for col in columns_for_svd_unique:
            if unique_movies_df[col].isnull().any():
                median_val = unique_movies_df[col].median()
                unique_movies_df[col] = unique_movies_df[col].fillna(median_val)
<<<<<<< HEAD
=======
        # Extra: fill any remaining NaNs with 0 (safety for SVD)
        unique_movies_df[columns_for_svd_unique] = unique_movies_df[columns_for_svd_unique].fillna(0)
>>>>>>> 848b2a5 (almost finilaize)

        if 'production_companies_Warner Bros._mean_vote_average' in unique_movies_df.columns:
            unique_movies_df['production_companies_Warner Bros._mean_vote_average'] = unique_movies_df['production_companies_Warner Bros._mean_vote_average'].fillna(0)


<<<<<<< HEAD
        n_components = 150
=======
        n_components = 120
>>>>>>> 848b2a5 (almost finilaize)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_matrix_unique = svd.fit_transform(unique_movies_df[columns_for_svd_unique])
        svd_df_unique = pd.DataFrame(svd_matrix_unique, columns=[f'svd_{i+1}' for i in range(n_components)], index=unique_movies_df.index)
        columns_to_drop_after_svd_unique = [col for col in columns_for_svd_unique if col not in ['vote_average', 'vote_count']]
        self.unique_movies_reduced = unique_movies_df.drop(columns=columns_to_drop_after_svd_unique).copy()
        self.unique_movies_reduced = pd.concat([self.unique_movies_reduced, svd_df_unique], axis=1)

    def run_all(self):
<<<<<<< HEAD
        self.ordering()
=======
>>>>>>> 848b2a5 (almost finilaize)
        self.outliers()
        self.add_budget_to_revenue_ratio()
        self.add_top_genre_onehot()
        self.add_log_features()
        self.add_interaction_features()
        self.add_count_features()
<<<<<<< HEAD
        self.add_text_length_features()
        self.add_genre_mean_encoding()
=======
>>>>>>> 848b2a5 (almost finilaize)
        self.add_release_date_features()
        self.add_adult_flag()
        self.add_multi_hot_keywords()
        self.add_cast_crew_features()
        self.add_company_country_features()
<<<<<<< HEAD
=======
        self.status_onehot()
>>>>>>> 848b2a5 (almost finilaize)
        self.coding()
        self.Tfidf()
        self.merging_Tfidf()
        self.presvd()
        self.svd()
        
        return {
            "merged_df": self.merged_df,
            "merged_df_with_tfidf": self.merged_df_with_tfidf,
            "unique_movies_reduced": self.unique_movies_reduced
        }
