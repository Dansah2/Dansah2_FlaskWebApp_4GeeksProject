from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
with open("/workspaces/Dansah2_FlaskWebApp_machine-learning-python-template/models/knn_neighbors-7_algorithm-auto_metric-cosine_leaf_size-40_radius-1.0.sav", "rb") as f:
    model = pickle.load(f)

#load the dataset
movies_df = pd.read_csv("/workspaces/Dansah2_FlaskWebApp_machine-learning-python-template/data/processed/processed_data.csv")  # Ensure this contains a 'title' column
movie_titles = movies_df["title"].tolist()

@app.route("/", methods = ["GET", "POST"])
def index():
    recommendations = None

    if request.method == "POST":
        movie_title = request.form["val1"]

        if movie_title in movie_titles:
            # Get the index of the movie
            movie_index = movie_titles.index(movie_title)
            
            # Find the nearest neighbors
            distances, indices = model.kneighbors([[movie_index]], n_neighbors=6)
            
            # Get recommended movie titles (excluding the first one, which is the input movie)
            recommendations = [movie_titles[i] for i in indices[0][1:]]

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)