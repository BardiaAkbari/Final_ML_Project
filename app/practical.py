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
import pandas as pd

def main():
    # Step 1: Preprocessing
    preprocessor = Preprocessing()
    dfs = preprocessor.run_all()

    # Step 2: EDA (plots and images will be saved)
    eda = EDA(dfs)
    eda.run_all()

    # Step 3: Feature Engineering
    fe = FeatureEngineering(dfs)
    fe.run_all()

    # Step 4: Modeling (baseline regression)
    modeling = Modeling()
    modeling.run_all()

    # Step 5: Content-Based Recommendation
    cb = ContentBasedRecommender()
    cb.build_item_vectors()
    # Example: recommend for a user (user_id=1)
    recommendations = cb.recommend(user_id=1, top_n=10)
    print("Content-based recommendations for user 1:")
    print(recommendations)
    # Example: explanation for a recommendation
    if not recommendations.empty:
        explanation = cb.explain(user_id=1, item_id=recommendations.iloc[0]['movieId'])
        print("Explanation for top recommendation:", explanation)

    # Step 6: Collaborative Filtering
    cf = CollaborativeFiltering()
    cf.run_all()

    # Step 7: Hybrid Model
    hybrid = HybridRecommender()
    hybrid.run_all(user_id=1, top_n=10)

if __name__ == "__main__":
    main()