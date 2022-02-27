import os
import git
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Init Flask app
#app = Flask(__name__)
app = Flask(__name__)

# Load model objects
my_directory = os.path.dirname(__file__)
pickle_model_objects_path = os.path.join(my_directory, "model_objects.pkl")
with open(pickle_model_objects_path, "rb") as handle:
    transformer, classifier = pickle.load(handle)


@app.route('/git_update', methods=['POST'])
def git_update():
    if request.method == 'POST':
        repo = git.Repo('./OpenClassrooms-P7')
        origin = repo.remotes.origin
        repo.create_head('main',origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
        origin.pull()
        return '', 200
    else:
        return '', 400



@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "GET":
        return "Prediction page"

    if request.method == "POST":
        # Parse data as JSON
        client_input = request.get_json()
        # Convert dictionary to pandas dataframe
        client_input = pd.DataFrame(client_input)
        # Transforming features
        client_input = transformer.transform(client_input)
        # Making predictions
        pred = classifier.predict(client_input)[0]
        proba = classifier.predict_proba(client_input)[0][pred]
        return jsonify(prediction=int(pred), probability=round(100 * proba, 1))

from flask import Flask
from flask.templating import render_template




@app.route('/')
def index():
    return render_template('index.html', name='home')




if __name__ == '__main__':
    app.run(debug=True)
