from kafka import KafkaConsumer
import json
import joblib
from subprocess import run
from prediction_model import preprocess_text

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

consumer = KafkaConsumer(
    'movie-reviews',
    bootstrap_servers='127.0.0.1:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='review-group', 
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
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
review_batch=[]
batch_size=10
for msg in consumer:
    review = msg.value['review']
    print(f"Received review: {review[:60]}")
    review_batch.append(review)
    if len(review_batch)==batch_size:
        recommendation=get_recommendation(review_batch)
        print(f"Batch of {batch_size} reviews → Recommendation: {recommendation}\n")
        review_batch=[]

