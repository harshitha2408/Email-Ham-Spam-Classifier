from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
nb_classifier = joblib.load('nb_classifier.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess text
def transform_text(text):
    # Implement your text preprocessing steps here
    return text

# Function to predict spam or ham
def predict_spam_or_ham(text):
    processed_text = transform_text(text)
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    prediction = nb_classifier.predict(text_vectorized)
    return "spam" if prediction[0] == 1 else "ham"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_spam_or_ham(text)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
