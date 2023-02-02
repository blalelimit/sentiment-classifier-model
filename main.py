from scripts.process import *


def classify_all(sentiment, model_name):
    # process
    hashtags = get_hashtags()
    vocabulary = get_vocabulary()
    sentiment_cleaned = Process(sentiment)
    sentiment_cleaned.preprocess_list(hashtags)
    sentiment_vectorized = sentiment_cleaned.vectorize_list(vocabulary)

    if len(sentiment_vectorized) != 0:
        return predict_all(sentiment_vectorized, model_name)
    else:
        print('Invalid input or your sentiment contains unrecognized text')
        return


if __name__ == '__main__':
    sentiment_main = input_sentiment()
    model = input_model()
    classify_all(sentiment_main, model)
