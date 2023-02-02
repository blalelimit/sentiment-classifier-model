# sentiment-classifier-model
COVID-19 Sentiment Classifier Model

# Overview
* A Twitter-based clustering and classification model able to determine the sentiment polarity (Positive, Neutral, Negative) of Tweets.
* The clustering model implemented is based on the K-Means method.
* The classifier models are Supervised Machine Learning algorithms consisting of Naïve Bayes, Support Vector Machine, Logistic Regression and Combination Ensemble (built from the aforementioned individual classifiers).

# Requirements
* Python 3.7
* To run main.py and webapp.py, the requirements are installed through:
```sh
  python -m pip install -r requirements.txt
```
* To train the model and run the main_system.ipynb, the requirements are installed through:
```sh
  python -m pip install -r requirements_training.txt
```

# Files
* main.py -> Python file that accepts sentiment inputs and returns the predicted sentiment polarity (Positive, Neutral, Negative) as classified by the classifier models.
* webapp.py -> Web application utilizing the Flask framework. Likewise accepts the sentiment inputs and returns the predicted
sentiment polarity.
* main_system.ipynb -> Jupyter Notebook file documenting the project. Includes the clustering and classification of inputs, as well as training the system.