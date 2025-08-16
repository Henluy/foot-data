import joblib
import pandas as pd
import argparse
import os

def make_prediction(home_form, away_form):
    """
    Loads the trained model and makes a prediction for a single match.
    Uses absolute paths to be robust to call location.
    """
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Define absolute paths for model and encoder
    model_path = os.path.join(project_root, 'models', 'model.joblib')
    encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')

    # Load the model and encoder
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except FileNotFoundError:
        print("Error: Model or label encoder not found. Please run train.py first.")
        return None, None

    # Create a DataFrame for the input features
    features = pd.DataFrame([[home_form, away_form]], columns=['HomeTeam_Form', 'AwayTeam_Form'])

    # Make prediction
    prediction_encoded = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Decode the prediction
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Get probabilities for each class
    probabilities = {label: prob for label, prob in zip(label_encoder.classes_, prediction_proba[0])}

    return prediction_label, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a football match outcome based on team form.")
    parser.add_argument("--home_form", type=float, required=True, help="Form points of the home team (e.g., 10).")
    parser.add_argument("--away_form", type=float, required=True, help="Form points of the away team (e.g., 8).")

    args = parser.parse_args()

    predicted_outcome, probabilities = make_prediction(args.home_form, args.away_form)

    if predicted_outcome:
        print("\n--- Match Prediction ---")
        print(f"Predicted Outcome: {predicted_outcome}")
        print("Prediction Probabilities:")
        for outcome, prob in probabilities.items():
            print(f"  - {outcome} (Win): {prob:.2%}")
        print("----------------------\n")
