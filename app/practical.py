import sys
import os

# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import Preprocessing
from src.eda import EDA

def main():
    # Step 1: Preprocessing
    preprocessor = Preprocessing()
    dfs = preprocessor.run_all()

    # Step 2: EDA (plots and images will be saved)
    eda = EDA(dfs)
    eda.run_all()

if __name__ == "__main__":
    print("START")
    main()
    print("END")