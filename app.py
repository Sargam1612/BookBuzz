from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Example dataset
df = pd.DataFrame({
    'book_title': [
        'Harry Potter', 
        'Game of Thrones', 
        'Lord of Rings', 
        'Name of the Wind'
    ],
    'author': ['Rowling', 'Martin', 'Tolkien', 'Rothfuss'],
    'genre': ['Fantasy', 'Fantasy', 'Fantasy', 'Fantasy']
})

# Mapping of titles to image filenames
IMAGE_MAP = {
    "harry potter": "harry_potter.jpg",
    "game of thrones": "game_of_thrones.jpg",
    "lord of rings": "lord_of_rings.jpg",
    "name of the wind": "name_of_the_wind.jpg"
}

# Simple content-based recommendation
df['features'] = df['author'] + " " + df['genre']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_books(title, n=4):
    if title not in df['book_title'].values:
        return []
    idx = df[df['book_title'] == title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    recommended = [df.iloc[i[0]]['book_title'] for i in sim_scores[1:n+1]]
    return recommended

@app.route('/', methods=['GET', 'POST'])
def home():
    books = df['book_title'].unique()
    recommendations = []
    selected_book = None
    if request.method == 'POST':
        selected_book = request.form.get('book')
        recs = recommend_books(selected_book)
        recommendations = [(rec, IMAGE_MAP.get(rec.lower(), None)) for rec in recs]
    return render_template('index.html', books=books, recommendations=recommendations, selected_book=selected_book)

if __name__ == '__main__':
    app.run(debug=True)
