import sys
import os

# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessing
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.modeling import RecommenderModels

def main():
    print("========== Step 1: Preprocessing ==========")
    preprocessor = Preprocessing()
    dfs = preprocessor.run_all()

    # print("========== Step 2: Exploratory Data Analysis (EDA) ==========")
    # eda = EDA(dfs)
    # eda.run_all()

    print("========== Step 3: Feature Engineering ==========")
    fe = FeatureEngineering(dfs)
    fe_outputs = fe.run_all()
    merged_df = fe_outputs["merged_df"]
    merged_df_with_tfidf = fe_outputs["merged_df_with_tfidf"]
    unique_movies_reduced = fe_outputs["unique_movies_reduced"]
    ratings_df = dfs["ratings_df"]

    print("========== Step 4: Modeling & Recommendation ==========")
    models = RecommenderModels(
        merged_df_with_tfidf=merged_df, 
        merged_df_with_tfidf=merged_df_with_tfidf,
        unique_movies_reduced=unique_movies_reduced, 
        ratings_df=ratings_df
    )
    models.fit_popularity()
    models.fit_content_based()
    models.fit_cf()
    print("CF RMSEs (kNN, SVD):", models.evaluate_cf())
    rmse_scores, best_alpha = models.tune_hybrid_alpha()
    print("Best alpha:", best_alpha)
    print("Hybrid RMSE:", models.evaluate_hybrid())

    # Example: get recommendations for user 1
    print("Top 10 Content-Based Recommendations for user 1:")
    print(models.get_content_based_recommendations(user_id=1, top_n=10))

    

if __name__ == "__main__":
    main()