from kafka import KafkaProducer
import json
import pandas as pd
import time

producer = KafkaProducer(
    bootstrap_servers='127.0.0.1:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

df = pd.read_csv("IMDB Dataset.csv")

for review in df['review']:
    producer.send("movie-reviews", {"review": review})
    print("Sent:", review[:60])
    time.sleep(0.5)  # simulate streaming

producer.flush()
producer.close()
