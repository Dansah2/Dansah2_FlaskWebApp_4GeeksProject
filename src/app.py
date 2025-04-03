from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained KNN model

model_path = "../models/knn_neighbors-7_algorithm-auto_metric-cosine_leaf_size-40_radius-1.0.sav"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the dataset
movies_df = pd.read_csv("../data/processed/processed_data.csv") 
movie_titles = movies_df["title"].tolist()

# Vectorize movie tags
vector = TfidfVectorizer()
matrix = vector.fit_transform(movies_df["tags"])

# Normalize titles for matching
movie_titles_lower = [title.strip().lower() for title in movie_titles]

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    error = None

    if request.method == "POST":
        movie_title = request.form["val1"].strip().strip('"').strip("'").lower()

        if movie_title in movie_titles_lower:
            # Get original index (before lowercasing)
            matched_index = movies_df[movies_df["title"].str.strip().str.lower() == movie_title].index
            
            if len(matched_index) == 0:
                error = "Movie not found. Please try another title."
            else:
                movie_index = matched_index[0]

                # Find the nearest neighbors
                distances, indices = model.kneighbors(matrix[movie_index])

                # Get recommended movie titles (excluding the first one, which is the input movie)
                recommendations = [movies_df.iloc[i]["title"] for i in indices[0][1:]]
        else:
            error = "Movie not found. Please try another title."

    return render_template("index.html", recommendations=recommendations, error=error)

if __name__ == "__main__":
    app.run(debug=True)

# Render Link: https://dansah2-flaskwebapp-machine-learning.onrender.com/