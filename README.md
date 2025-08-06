# RECOMMENDATION-SYSTEMimport 
pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'title': [
        'The Matrix', 'John Wick', 'Inception',
        'The Notebook', 'Interstellar', 'Avengers: Endgame',
        'Titanic', 'The Dark Knight', 'Gravity', 'The Godfather'
    ],
    'description': [
        'sci-fi action reality hacking',
        'action assassin revenge dog',
        'dream sci-fi thriller mind-bending',
        'romance drama notebook love',
        'space sci-fi drama time travel',
        'superhero marvel action time travel',
        'romance drama ship ocean',
        'dark hero crime vigilante',
        'space astronaut survival drama',
        'mafia crime drama classic'
    ]
}

# Load into DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, df, cosine_sim):
    if title not in df['title'].values:
        print("Movie not found in the database.")
        return
    
    # Get index of the movie
    idx = df.index[df['title'] == title][0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity (excluding itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    # Get movie titles
    movie_indices = [i[0] for i in sim_scores]
    
    print(f"\nBecause you liked '{title}', you might also like:")
    for i in movie_indices:
        print(f"- {df['title'][i]}")

# Try it out
get_recommendations('Inception', df, cosine_sim)
