# importing libraries
from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

application = Flask(__name__)
@application.route('/')

def home():
    return render_template("main.html")

@application.route("/predict", methods = ["POST"])

def predict():
    # loading the dataset
    data = pd.read_csv(r"Language_Detection.csv")
    y = data["Language"]

    # label encoding
    y = le.fit_transform(y)

    #loading the model and cv
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))

    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # preprocessing the text
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        # creating the vector
        x = cv.transform(dat).toarray()
        # prediction
        probabilities = model.predict_proba(x)
        max_prob = probabilities.max()
        threshold=0.75
        if max_prob < threshold:
            predicted_language = "Unknown"
        else:
            lang_idx = probabilities.argmax()
            predicted_language = le.inverse_transform([lang_idx])[0]

    return render_template("main.html", pred="This word/sentence contains {} word(s).".format(predicted_language))


if __name__ =="__main__":
    application.run(debug=True)
