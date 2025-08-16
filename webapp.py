import sys
import os
import json
from flask import Flask, render_template, request, url_for
import joblib
import pandas as pd

app = Flask(__name__)

def load_model_and_predict(home_form, away_form):
    """
    Loads the trained model and makes a prediction for a single match.
    """
    project_root = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(project_root, 'models', 'model.joblib')
    encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')

    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except FileNotFoundError:
        return None, None

    features = pd.DataFrame([[home_form, away_form]], columns=['HomeTeam_Form', 'AwayTeam_Form'])
    prediction_encoded = model.predict(features)
    prediction_proba = model.predict_proba(features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    probabilities = {label: prob for label, prob in zip(label_encoder.classes_, prediction_proba[0])}

    return prediction_label, probabilities

@app.route('/')
def dashboard():
    """
    Renders the main dashboard, loading model metrics and sample data.
    """
    project_root = os.path.abspath(os.path.dirname(__file__))
    metrics_path = os.path.join(project_root, 'models', 'metrics.json')
    data_path = os.path.join(project_root, 'data', 'processed_data.csv')
    cm_image_path = 'static/confusion_matrix.png'

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    data_sample = None
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        data_sample = df.head(10).to_html(classes='data-table', index=False)

    cm_image_exists = os.path.exists(cm_image_path)

    return render_template('dashboard.html',
                           metrics=metrics,
                           data_sample=data_sample,
                           cm_image_url=url_for('static', filename='confusion_matrix.png') if cm_image_exists else None)

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handles form submission and renders the result."""
    try:
        home_form = float(request.form['home_form'])
        away_form = float(request.form['away_form'])

        predicted_label, probabilities = load_model_and_predict(home_form, away_form)

        if predicted_label is None:
            return "Erreur : Le modèle n'a pas pu être chargé.", 500

        if predicted_label == 'H':
            prediction_text = "Victoire à Domicile"
        elif predicted_label == 'A':
            prediction_text = "Victoire à l'Extérieur"
        else:
            prediction_text = "Match Nul"

        return render_template('result.html',
                               prediction=prediction_text,
                               probabilities=probabilities)

    except (ValueError, KeyError):
        return "Erreur : Données d'entrée invalides.", 400
    except Exception as e:
        return f"Une erreur inattendue est survenue : {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
