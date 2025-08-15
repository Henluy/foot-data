import pandas as pd
import joblib
import os
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, encoder_path, metrics_output_path):
    """
    Loads a specific model and encoder, evaluates performance,
    and saves the key metrics to a JSON file.
    """
    print(f"Starting evaluation for model: {model_path}")

    # Define paths
    test_X_path = os.path.join('data', 'test_set_X.csv')
    test_y_path = os.path.join('data', 'test_set_y.csv')
    cm_plot_path = os.path.join(os.path.dirname(model_path), f"cm_{os.path.basename(model_path).split('.')[0]}.png")

    # Load model, encoder, and test data
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        X_test = pd.read_csv(test_X_path)
        y_test = pd.read_csv(test_y_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please run train.py first.")
        return

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # --- Metrics Calculation ---
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    # --- Save Metrics to JSON ---
    metrics = {
        'accuracy': accuracy,
        'log_loss': loss,
        'classification_report': report
    }
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_output_path}")

    # --- Display Results ---
    print("\n--- Model Evaluation Report ---")
    print(f"\nLog-Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("---------------------------------")

    # --- Save Confusion Matrix Plot ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {os.path.basename(model_path)}')
    plt.ylabel('Actual Result')
    plt.xlabel('Predicted Result')
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.joblib",
        help="Path to the trained model file."
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default="models/label_encoder.joblib",
        help="Path to the label encoder file."
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="models/metrics.json",
        help="Path to save the output metrics JSON file."
    )
    args = parser.parse_args()

    try:
        evaluate_model(args.model_path, args.encoder_path, args.metrics_path)
    except ImportError:
        print("\nPlease install seaborn and matplotlib to generate the confusion matrix plot:")
        print("pip install seaborn matplotlib")
    except Exception as e:
        print(f"An error occurred: {e}")
