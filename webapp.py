from flask import Flask, request, render_template
from scripts.model import *

# Create flask app
app = Flask(__name__)


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    if request.method == "POST":
        sentiment = request.form.get("sentiment")
        model = request.form.get("model")
        cluster = request.form.get("cluster")
        text = 'Your sentiment is: ' + sentiment

        if sentiment == '' or model == '' or cluster == '':
            output = 'Invalid input'

        else:
            sentiment = pd.DataFrame([sentiment], columns={'text'})
            prediction = classify(sentiment, model, cluster)
            output = prediction

    else:
        output = ''
        text = ''

    return render_template('index.html', output=output, text=text)


if __name__ == '__main__':
    app.run(debug=True)
