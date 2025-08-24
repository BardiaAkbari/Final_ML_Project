# MovieLens Movie Data Analysis

This project provides a reproducible pipeline for preprocessing and exploratory data analysis (EDA) on the MovieLens movie dataset.

## Project Structure

```
.
├── app/
│   └── Practical.py         # Main entry point for running the pipeline
├── src/
│   ├── preprocessing.py     # Data loading, cleaning, merging
│   └── eda.py               # EDA and visualization (plots saved to /report/images)
├── notebooks/
│   └── Practical.ipynb      # Step-by-step notebook for exploration and prototyping
├── report/
│   └── images/              # Output directory for all generated plots and images
├── data/
│   ├── raw/                 # Raw input data (CSV files)
│   ├── interim/             # Cleaned/intermediate CSVs
│   └── processed/           # (Optional) Final processed data
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## How to Run

1. **Install dependencies**  
   Make sure you have Python 3.8+ and run:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Place the raw MovieLens CSV files in `data/raw/` as:
   - `movies_metadata.csv`
   - `credits.csv`
   - `keywords.csv`
   - `links.csv`
   - `ratings.csv`

3. **Run the pipeline**
   ```
   python app/Practical.py
   ```
   This will:
   - Clean and merge the data
   - Save interim cleaned CSVs to `data/interim/`
   - Generate all EDA plots and wordclouds, saving them to `report/images/`
   - Save interactive Plotly plots as PNG (requires [kaleido](https://github.com/plotly/Kaleido)) or HTML fallback

## Features

- **Modular Preprocessing**: All data cleaning, merging, and type handling in `src/preprocessing.py`
- **Automated EDA**: All plots and wordclouds generated and saved by `src/eda.py`
- **Reproducibility**: One-command run for the entire workflow
- **Notebook**: `notebooks/Practical.ipynb` for step-by-step exploration

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- missingno
- wordcloud
- plotly
- pycountry
- kaleido (for static plotly image export)

## Notes

- If static Plotly image export fails, HTML versions of the plots are saved as a fallback.
- All output images are saved in `report/images/`.
- Adjust paths in `src/eda.py` and `src/preprocessing.py` if your