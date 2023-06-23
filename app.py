from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
import requests

app = Flask(__name__)

# Load the movie data
movies_credits = pd.read_csv('movies_credits.csv')

# Compute the cosine similarity matrix
cosine_sim2 = cosine_similarity(tfidfv_matrix, tfidfv_matrix)

# Set up the Surprise library
reader = Reader()
ratings = pd.read_csv('ratings.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Create a dictionary mapping movie titles to their indices
indices = pd.Series(movies_credits.index, index=movies_credits['title']).drop_duplicates()

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get the user ID and movie title from the request
    user_id = int(request.json['userId'])
    movie_title = request.json['title']

    # Get the index of the movie that matches the title
    idx = indices[movie_title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [x[0] for x in sim_scores]

    # Grab the title, movie ID, vote average, and vote count of the top 10 most similar movies
    recommendations = movies_credits.iloc[movie_indices][['title', 'movieId', 'vote_average', 'vote_count']]

    # Predict the ratings a user might give to these top 10 most similar movies
    estimated_ratings = []
    for movie_id in recommendations['movieId']:
        estimated_ratings.append(algo.predict(user_id, movie_id).est)

    # Add the estimated ratings to the recommendations DataFrame
    recommendations['estimated_rating'] = estimated_ratings

    # Get the poster images for the recommendations from TMDb API
    api_key = '53607277a4abba625e13562a61ea99d5'
    base_url = 'https://api.themoviedb.org/3/movie/'
    poster_size = 'w500'
    posters = []
    for movie_id in recommendations['movieId']:
        url = f'{base_url}{movie_id}?api_key={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                poster_path = data['poster_path']
                poster_url = f'https://image.tmdb.org/t/p/{poster_size}/{poster_path}'
                posters.append(poster_url)
            else:
                posters.append(None)
        else:
            posters.append(None)

    # Add the poster URLs to the recommendations DataFrame
    recommendations['poster_url'] = posters

    # Return the recommendations as JSON response
    return recommendations.to_json(orient='records')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

flask run
