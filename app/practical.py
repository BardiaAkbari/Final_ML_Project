import sys
import os

# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessing
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.modeling import RecommenderModels
<<<<<<< HEAD
=======
from src.evaluation import leave_one_out_by_timestamp, evaluate_all, summarize_results
>>>>>>> 848b2a5 (almost finilaize)

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
<<<<<<< HEAD
    merged_df = fe_outputs["merged_df"]
=======
>>>>>>> 848b2a5 (almost finilaize)
    merged_df_with_tfidf = fe_outputs["merged_df_with_tfidf"]
    unique_movies_reduced = fe_outputs["unique_movies_reduced"]
    ratings_df = dfs["ratings_df"]

    print("========== Step 4: Modeling & Recommendation ==========")
    models = RecommenderModels(
<<<<<<< HEAD
        merged_df_with_tfidf=merged_df, 
=======
>>>>>>> 848b2a5 (almost finilaize)
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
<<<<<<< HEAD

=======
    models.save_models()
>>>>>>> 848b2a5 (almost finilaize)
    # Example: get recommendations for user 1
    print("Top 10 Content-Based Recommendations for user 1:")
    print(models.get_content_based_recommendations(user_id=1, top_n=10))

<<<<<<< HEAD
    
=======
    print("========== Step 5: Evaluation ==========")
    train_ratings, test_ratings = leave_one_out_by_timestamp(ratings_df)
    all_items = set(merged_df_with_tfidf['movieId'].astype(str).unique())
    item_popularity = merged_df_with_tfidf['movieId'].value_counts().to_dict()
    item_popularity = {str(k): v for k, v in item_popularity.items()}
    svd_cols = [col for col in unique_movies_reduced.columns if col.startswith("svd_")]
    item_features = {
        str(row.movieId): row[svd_cols].values
        for _, row in unique_movies_reduced.iterrows()
    }

    def predict_content_based(models, test_df):
        preds = []
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']
            pred_rating = models.get_content_based_score(user_id, movie_id)
            preds.append((user_id, movie_id, true_rating, pred_rating, {}))
        return preds

    def predict_collaborative(models, test_df):
        preds = []
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']
            try:
                pred_rating = models.svd_mf.predict(str(user_id), str(movie_id)).est
            except Exception:
                pred_rating = 0
            preds.append((user_id, movie_id, true_rating, pred_rating, {}))
        return preds

    def predict_hybrid(models, test_df, alpha):
        preds = []
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']
            pred_rating = models.hybrid_prediction(user_id, movie_id, alpha)
            preds.append((user_id, movie_id, true_rating, pred_rating, {}))
        return preds

    predictions_cb = predict_content_based(models, test_ratings)
    predictions_cf = predict_collaborative(models, test_ratings)
    predictions_hybrid = predict_hybrid(models, test_ratings, alpha=best_alpha)

    results_cb = evaluate_all(predictions_cb, test_ratings.values, all_items, item_popularity, item_features)
    results_cf = evaluate_all(predictions_cf, test_ratings.values, all_items, item_popularity, item_features)
    results_hybrid = evaluate_all(predictions_hybrid, test_ratings.values, all_items, item_popularity, item_features)

    summary = summarize_results({
        "Content-Based": results_cb,
        "Collaborative": results_cf,
        "Hybrid": results_hybrid
    })
    print(summary)

>>>>>>> 848b2a5 (almost finilaize)

if __name__ == "__main__":
    main()