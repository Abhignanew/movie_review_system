import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# Load the dataset
df = pd.read_csv(r"IMDB Dataset.csv")


# Preprocess the text data
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

df['review'] = df['review'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Recommendation logic function
def get_recommendation(reviews):
    positive_count = 0
    negative_count = 0
    for review in reviews:
        review_processed = preprocess_text(review)
        review_tfidf = vectorizer.transform([review_processed])
        prediction = model.predict(review_tfidf)[0]
        if prediction == 'positive':
            positive_count += 1
        else:
            negative_count += 1
    print(f"positive {positive_count} negative {negative_count}")

    if positive_count > negative_count:
        return "Highly Recommended"
    elif positive_count == negative_count:
        return "Average"
    else:
        return "Not Recommended"

# Example usage with some reviews for a hypothetical movie
movie_reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "A complete waste of time. The plot was predictable and the acting was terrible.",
    "A decent movie, but nothing special.",
    "I would definitely watch this again. Great performances and a compelling story.",
    "I'm not sure why this movie gets so much hype. I found it to be boring."
]

recommendation = get_recommendation(movie_reviews)
print(f"Movie Recommendation: {recommendation}")

import joblib
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
