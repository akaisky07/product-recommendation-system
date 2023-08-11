from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def get_top_similar_products(input_string, df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
    input_vector = tfidf_vectorizer.transform([input_string])

    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
    similar_indices = cosine_similarities.argsort()[::-1]

    top_similar_indices = similar_indices[:10]
    top_similar_products = df.iloc[top_similar_indices]['product_name'].tolist()

    return top_similar_products

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_string = request.form['input_string']

        df = pd.read_csv('your_dataset.csv')
        df['description'].fillna('', inplace=True)

        p_names = get_top_similar_products(input_string, df)

        return render_template('index.html', p_names=p_names)

    return render_template('index.html', p_names=None)

if __name__ == '__main__':
    app.run(debug=True)

