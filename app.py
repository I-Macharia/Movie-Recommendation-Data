import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    api_key = "53607277a4abba625e13562a61ea99d5"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def hybrid_recommendations(title, cosine_sim2, movies_credits):
    # Get the index of the movie that matches the title
    idx = movies_credits[movies_credits['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    ind = [x for x, _ in sim_scores]

    # Grab the title, id, vote_average, and vote_count of the top 10 most similar movies
    tit = []
    movieid = []
    vote_average = []
    vote_count = []
    for x in ind:
        tit.append(movies_credits.iloc[x]['title'])
        movieid.append(movies_credits.iloc[x]['id'])
        vote_average.append(movies_credits.iloc[x]['vote_average'])
        vote_count.append(movies_credits.iloc[x]['vote_count'])

    return pd.DataFrame({
        'index': ind,
        'title': tit,
        'id': movieid,
        'vote_average': vote_average,
        'vote_count': vote_count
    }).set_index('index').sort_values(by='vote_average', ascending=False)


st.header('Movie Recommender System')

# Absolute paths to the data files
tmdb_movies_path = r".data\tmdb_5000_movies.csv"
tmdb_credits_path = r".data\tmdb_5000_credits.csv"

# Read the CSV files
tmdb_movies = pd.read_csv(tmdb_movies_path)
tmdb_credits = pd.read_csv(tmdb_credits_path)

# Drop the Title column in Movies Dataset
tmdb_movies.drop(['title'], axis=1, inplace=True)

# Identify the columns that are common and need to be merged
tmdb_credits.columns = ['id', 'title', 'cast', 'crew']

# Adjust column names if necessary
movies_credits = pd.merge(tmdb_credits, tmdb_movies, on='id')

# Create the 'soup' column by combining relevant features
movies_credits['soup'] = movies_credits['overview'] + ' ' + movies_credits['genres'] + ' ' + movies_credits['keywords']

# Fill missing values in 'soup' column with an empty string
movies_credits['soup'].fillna('', inplace=True)

# Calculate cosine similarity
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(movies_credits['soup'])
cosine_sim2 = cosine_similarity(cv_matrix, cv_matrix)

movie_list = movies_credits['title'].values

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movies = hybrid_recommendations(selected_movie, cosine_sim2, movies_credits)

    columns = st.columns(5)
    displayed_titles = []
    for _, movie in recommended_movies.iterrows():
        title = movie['title']
        if title not in displayed_titles:
            displayed_titles.append(title)
            with columns[0]:
                st.text(title)
                st.image(fetch_poster(movie['id']))
            columns = columns[1:] + [columns[0]]
