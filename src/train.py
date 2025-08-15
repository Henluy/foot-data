import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import argparse

def train_model(model_output_path, encoder_output_path):
    """
    Loads processed data, trains a logistic regression model,
    and saves the model and encoder to the specified paths.
    It also saves the test set for later evaluation.
    """
    print("Starting model training...")
    processed_data_path = 'data/processed_data.csv'

    # Define paths for saving the test set
    test_X_path = os.path.join('data', 'test_set_X.csv')
    test_y_path = os.path.join('data', 'test_set_y.csv')

    # Create parent directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    df.dropna(subset=['HomeTeam_Form', 'AwayTeam_Form', 'FTR'], inplace=True)

    features = ['HomeTeam_Form', 'AwayTeam_Form']
    target = 'FTR'

    X = df[features]
    y = df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on the test set: {accuracy:.4f}")

    # Save the model and the encoder to the specified paths
    joblib.dump(model, model_output_path)
    joblib.dump(label_encoder, encoder_output_path)
    print(f"Model saved to {model_output_path} and encoder to {encoder_output_path}")

    # Save the test set for external evaluation
    X_test.to_csv(test_X_path, index=False)
    pd.DataFrame(y_test, columns=['FTR_encoded']).to_csv(test_y_path, index=False)
    print(f"Test data saved to {test_X_path} and {test_y_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and save it.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="models/label_encoder.joblib",
        help="Path to save the label encoder."
    )
    args = parser.parse_args()

    train_model(args.model_path, args.encoder_path)
