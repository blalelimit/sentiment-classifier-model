from scripts.process import *


def classify(sentiment, model_name, cluster):
    # process
    hashtags = get_hashtags()
    vocabulary = get_vocabulary()
    sentiment_cleaned = Process(sentiment)
    sentiment_cleaned.preprocess_list(hashtags)
    sentiment_vectorized = sentiment_cleaned.vectorize_list(vocabulary)

    if len(sentiment_vectorized) != 0:
        return predict_list(sentiment_vectorized, model_name, cluster)
    else:
        return 'Invalid input or your sentiment contains unrecognized text'
