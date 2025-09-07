import gradio as gr
import joblib
import pandas as pd
import os


MODEL_DIR = "models"
MOVIE_DATA_PATH = "data/raw/movies_metadata.csv"

with open(os.path.join(MODEL_DIR, "recommender_svd_mf.pkl"), "rb") as f:
    svd_model = joblib.load(f)

movies_df = pd.read_csv(MOVIE_DATA_PATH, low_memory=False)
movies_df["movieId"] = movies_df["movieId"].astype(str)

def recommend(user_id, top_k=5):
    """Generate top-k recommendations using SVD model."""
    all_movie_ids = movies_df["movieId"].unique()
    predictions = []
    for mid in all_movie_ids:
        try:
            est = svd_model.predict(str(user_id), str(mid)).est
            predictions.append((mid, est))
        except Exception:
            continue

    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for mid, score in top_movies:
        movie_rows = movies_df[movies_df["movieId"] == str(mid)]
        if not movie_rows.empty:
            row = movie_rows.iloc[0]
            explanation = f"Because you liked movies with {row.get('actors', 'similar style')}."
            results.append((row.get("title", "Unknown"), row.get("poster_url", None), explanation))
        else:
            results.append(("Unknown", None, "No explanation available."))
    
    return results

def format_output(results):
    titles = [r[0] for r in results]
    posters = [r[1] for r in results if r[1] is not None]
    explanations = [r[2] for r in results]
    return titles, posters, explanations

demo = gr.Interface(
    fn=lambda user_id, k: format_output(recommend(user_id, k)),
    inputs=[
        gr.Number(label="User ID"),
        gr.Slider(1, 10, value=5, step=1, label="Top-K")
    ],
    outputs=[
        gr.Textbox(label="Recommended Movies"),
        gr.Gallery(label="Posters"),
        gr.Textbox(label="Explanations")
    ],
    title="Movie Recommender System",
    description="Enter your User ID to get top-K movie recommendations with posters and explanations."
)

if __name__ == "__main__":
    demo.launch()
