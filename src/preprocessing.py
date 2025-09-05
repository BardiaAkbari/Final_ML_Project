import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import missingno as msno
import ast
warnings.filterwarnings('ignore')

class Preprocessing:
    def __init__(self):
        self.main_path = "D:/Uni/Term 6/Machine Learning/HomeWork/6/data/raw/"
        self.movies_metadata_path = self.main_path + "movies_metadata.csv"
        self.credits_path = self.main_path + "credits.csv"
        self.keywords_path = self.main_path + "keywords.csv"
        self.links_path = self.main_path + "links_small.csv"
        self.ratings_path = self.main_path + "ratings_small.csv"
        self.img_path = "D:/Uni/Term 6/Machine Learning/HomeWork/6/report/images/"
        self.interim_path = "D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/"
        self.proceed_path = "D:/Uni/Term 6/Machine Learning/HomeWork/6/data/processed/"

    def load_data(self):
        self.df = pd.read_csv(self.movies_metadata_path)
        self.credits_df = pd.read_csv(self.credits_path)
        self.keywords_df = pd.read_csv(self.keywords_path)
        self.links_df = pd.read_csv(self.links_path)
        self.ratings_df = pd.read_csv(self.ratings_path)

    def df_missing_value(self):
        # saving missing value pic
        ax = msno.matrix(self.df)
        fig = ax.get_figure()
        fig.savefig(self.img_path + "df_missing.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        # Analyze missing value percentages from missing_df_info
        missing_df = self.df.isnull().sum().sort_values(ascending=False)
        missing_df_percent = (missing_df / len(self.df)) * 100
        missing_df_info = pd.DataFrame({'Missing Count': missing_df, 'Missing Percentage (%)': missing_df_percent})
        
        high_missing_cols = missing_df_info[missing_df_info['Missing Percentage (%)'] > 50].index.tolist()
        moderate_missing_cols = missing_df_info[(missing_df_info['Missing Percentage (%)'] <= 50) & (missing_df_info['Missing Percentage (%)'] > 1)].index.tolist()
        low_missing_cols = missing_df_info[missing_df_info['Missing Percentage (%)'] <= 1].index.tolist()
        handling_strategy = {}

        for col in high_missing_cols:
            handling_strategy[col] = "Drop column due to high missing percentage"

        if 'overview' in moderate_missing_cols:
            handling_strategy['overview'] = "Consider dropping or using a placeholder for text data"

        numerical_low_missing = ['runtime', 'vote_average', 'vote_count', 'revenue', 'popularity']
        for col in low_missing_cols:
            if col in numerical_low_missing:
                handling_strategy[col] = "Impute with mean or median"
            elif col in ['status', 'release_date', 'imdb_id', 'original_language', 'title', 'video', 'spoken_languages', 'production_countries', 'production_companies', 'poster_path']:
                handling_strategy[col] = "Consider imputation (e.g., mode, placeholder) or dropping rows"


        handling_strategy['adult'] = "Investigate and potentially remove incorrect entries (e.g., 'R')"
        
        # 1. Drop columns with high missing percentages
        cols_to_drop = ['belongs_to_collection', 'homepage', 'tagline']
        self.df = self.df.drop(columns=cols_to_drop)

        # 2. Fill missing values in 'overview' with a placeholder
        self.df['overview'] = self.df['overview'].fillna('No overview available')

        self.df['popularity'] = pd.to_numeric(self.df['popularity'], errors='coerce')

        numerical_cols_to_impute = ['runtime', 'vote_average', 'vote_count', 'revenue', 'popularity']
        for col in numerical_cols_to_impute:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        remaining_missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        cols_to_fill_unknown = [col for col in remaining_missing_cols if col not in numerical_cols_to_impute and col != 'adult']

        for col in cols_to_fill_unknown:
            self.df[col] = self.df[col].fillna('Unknown')

        # Investigate and remove incorrect entries in 'adult' column
        self.df = self.df[self.df['adult'].isin(['True', 'False'])]

    def extract_names_and_handle_empty(self, json_list_string):
            """Extracts names from a string representation of a list of dictionaries and handles empty lists as NaN."""
            if isinstance(json_list_string, str) and json_list_string.startswith('[') and json_list_string.endswith(']'):
                try:
                    data_list = ast.literal_eval(json_list_string)
                    if isinstance(data_list, list):
                        if not data_list:
                            return np.nan
                        names = [item.get('name', '') for item in data_list if isinstance(item, dict) and 'name' in item]
                        return ', '.join(names)
                except (ValueError, SyntaxError):
                    pass
            return ''
        

    def extract_names_from_list(self,json_list_string, key='name'):
        """Extracts values for a given key from a string representation of a list of dictionaries and handles empty lists as NaN."""
        if isinstance(json_list_string, str) and json_list_string.startswith('[') and json_list_string.endswith(']'):
            try:
                data_list = ast.literal_eval(json_list_string)
                if isinstance(data_list, list):
                    if not data_list:
                        return np.nan
                    names = [item.get(key, '') for item in data_list if isinstance(item, dict) and key in item]
                    return ', '.join(names)
            except (ValueError, SyntaxError):
                pass
        return ''
    
    def clean_data(self):
        # Drop null values in links_df
        self.links_df = self.links_df.dropna(subset=['tmdbId'])
        # Handling data types
        self.df['id'] = pd.to_numeric(self.df['id'], errors='coerce')
        self.df.dropna(subset=['id'], inplace=True)
        self.df['id'] = self.df['id'].astype(int)

        self.links_df.dropna(subset=['tmdbId'], inplace=True)
        self.links_df['tmdbId'] = self.links_df['tmdbId'].astype(int)
    
        copy_df = self.df.copy()
        # Columns identified as containing JSON format
        json_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages']

        # Apply the extraction function and handle empty lists to copy_df
        for col in json_columns:
            if col in copy_df.columns:
                copy_df[col] = copy_df[col].apply(self.extract_names_and_handle_empty)
                # Now fill the NaN values (which were empty lists) with 'Unknown'
                copy_df[col] = copy_df[col].fillna('Unknown')
        self.df = copy_df.copy()

        # Handle JSON columns in credits_df
        # 'cast' and 'crew' columns contain lists of dictionaries, we'll extract names
        self.credits_df['cast'] = self.credits_df['cast'].apply(self.extract_names_from_list, key='name')
        self.credits_df['crew'] = self.credits_df['crew'].apply(self.extract_names_from_list, key='name')

        # Fill NaN values (originally empty lists) with 'Unknown' in credits_df
        self.credits_df['cast'] = self.credits_df['cast'].fillna('Unknown')
        self.credits_df['crew'] = self.credits_df['crew'].fillna('Unknown')

        # Handle JSON columns in keywords_df
        # 'keywords' column contains a list of dictionaries, we'll extract names
        self.keywords_df['keywords'] = self.keywords_df['keywords'].apply(self.extract_names_from_list, key='name')

        # Fill NaN values (originally empty lists) with 'Unknown' in self.keywords_df
        self.keywords_df['keywords'] = self.keywords_df['keywords'].fillna('Unknown')

        # Remove duplicates from key columns
        self.df.drop_duplicates(subset=['id'], inplace=True)
        self.credits_df.drop_duplicates(subset=['id'], inplace=True)
        self.keywords_df.drop_duplicates(subset=['id'], inplace=True)
        self.links_df.drop_duplicates(subset=['tmdbId'], inplace=True)
        self.ratings_df.drop_duplicates(subset=['movieId', 'userId'], inplace=True)
        

    def merge_data(self):
        # Merge df, credits_df, and keywords_df on 'id'
        self.merged_df = pd.merge(self.df, self.credits_df, on='id', how='inner')
        self.merged_df = pd.merge(self.merged_df, self.keywords_df, on='id', how='inner')
        # Merge with links_df using tmdbId from links_df and id from self.merged_df
        self.merged_df = pd.merge(self.merged_df, self.links_df, left_on='id', right_on='tmdbId', how='inner')
        # Do NOT merge with ratings_df here! Only merge for modeling step.

    def generate_interim_va_proceed_csv(self):
        # Save cleaned DataFrames to interim CSV files
        self.df.to_csv(self.interim_path + "movies_metadata_clean.csv", index=False)
        self.credits_df.to_csv(self.interim_path + "credits_clean.csv", index=False)
        self.keywords_df.to_csv(self.interim_path + "keywords_clean.csv", index=False)
        self.links_df.to_csv(self.interim_path + "links_clean.csv", index=False)
        self.ratings_df.to_csv(self.interim_path + "ratings_clean.csv", index=False)
        self.merged_df.to_csv(self.proceed_path + "merged_clean.csv", index=False)

    def run_all(self):
        self.load_data()
        self.df_missing_value()
        self.clean_data()
        self.merge_data()
        self.generate_interim_va_proceed_csv()
        return {
            "df": self.df,
            "credits_df": self.credits_df,
            "keywords_df": self.keywords_df,
            "links_df": self.links_df,
            "ratings_df": self.ratings_df,
            "merged_df": self.merged_df
        }