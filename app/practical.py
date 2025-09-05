import sys
import os

# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessing
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
from src.modeling import Modeling
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeFiltering
from src.hybrid import HybridRecommender

def main():
    print("========== Step 1: Preprocessing ==========")
    preprocessor = Preprocessing()
    dfs = preprocessor.run_all()

    # print("========== Step 2: Exploratory Data Analysis (EDA) ==========")
    # eda = EDA(dfs)
    # eda.run_all()

    print("========== Step 3: Feature Engineering ==========")
    fe = FeatureEngineering(dfs)
    fe.run_all()

    print("========== Step 4: Baseline Modeling (Regression) ==========")
    modeling = Modeling(feature_path="D:/Uni/Term 6/Machine Learning/HomeWork/6/data/interim/feature_engineered_with_ratings.csv")
    modeling.run_all()

    print("========== Step 5: Content-Based Recommendation ==========")
    cb = ContentBasedRecommender()
    cb.build_item_vectors()
    recommendations = cb.recommend(user_id=1, top_n=10)
    print("Content-based recommendations for user 1:")
    print(recommendations)
    if not recommendations.empty:
        explanation = cb.explain(user_id=1, item_id=recommendations.iloc[0]['movieId'])
        print("Explanation for top recommendation:", explanation)

    print("========== Step 6: Collaborative Filtering ==========")
    cf = CollaborativeFiltering()
    cf.run_all()

    print("========== Step 7: Hybrid Recommendation ==========")
    hybrid = HybridRecommender()
    hybrid.run_all(user_id=1, top_n=10)

if __name__ == "__main__":
    main()