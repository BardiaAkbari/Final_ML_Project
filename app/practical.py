import sys
import os

# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessing
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
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

if __name__ == "__main__":
    
    main()