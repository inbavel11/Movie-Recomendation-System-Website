from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load movie data
movies = pd.read_csv('E:/Projects/movie-recommendation/data/movies.csv')

# Use a smaller subset of the data for demonstration purposes
movies = movies.head(1000)

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies[movies['title'] == title].index[0]
    except IndexError:
        return ["Movie not found in the dataset."]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Dictionary mapping genres to image URLs
genre_image_paths = {
    "Action": "https://www.pinkvilla.com/english/images/2023/07/1941022096_orange-yellow-minimalist-aesthetic-a-day-in-my-life-travel-vlog-youtube-thumbnail_1280*720.jpg",
    "Adventure": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuWphUVFEpxDa-Hp_gBxiKIuXiMQT10bP_60pvlbmlxkjFf7nP_7t6Ly_r1swWAL4PN48&usqp=CAU",
    "Animation": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOmvxVinvIPTdPnLjoHvLkP_XQbyzK7teB-Q&s",
    "Children": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgc4S_cOISQeg_8IWJDF52GiNYPLA4lJFo0g&s",
    "Comedy": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxi0dQlPPYPIbEt3_ZeISzZL8jSA_95eCZXw&s",
    "Crime": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTj0V6cgt_mXKlct0KTPc81KiKUk8VxeTzidA&s",
    "Drama": "https://static1.moviewebimages.com/wordpress/wp-content/uploads/2023/07/the-20-best-fantasy-movies-of-all-time.jpg",
    "Fantasy": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRyBnCMO8rR3VdQDdOjF_zTyyCnEtU2LNqvBQ&s",
    "Horror": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTiYH58la4SFKAa9y9BvprpXec4l-53oX81rA&s",
    "Mystery": "https://variety.com/wp-content/uploads/2024/02/Best-Romance-Movies-for-Valentines-Day.jpg?w=1000&h=562&crop=1",
    "Romance": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2tFJaBqgfUw2ZM9BzeZOGky1wJ-3aP4_tBw&s",
    "Sci-Fi": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgVduYqnGsWKTqYNsZww8MGNJEviWn2cw9aA&s",
    "Thriller": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSgVduYqnGsWKTqYNsZww8MGNJEviWn2cw9aA&s",
    # Add more mappings here
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations = get_recommendations(title)
    return render_template('index.html', recommendations=recommendations, genre_image_paths=genre_image_paths, movies=movies)

if __name__ == '__main__':
    app.run(debug=True)