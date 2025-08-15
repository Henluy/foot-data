import argparse
import subprocess
import sys
import os
import json

def run_step(command):
    """Runs a command, checks for errors, and captures output."""
    print(f"\n--- Running step: {' '.join(command)} ---")
    process = subprocess.run(command, capture_output=True, text=True, check=False)

    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr, file=sys.stderr)

    if process.returncode != 0:
        print(f"--- Step failed with return code {process.returncode} ---")
        sys.exit(process.returncode)

    print(f"--- Step successful ---")

def run_full_pipeline():
    """Runs the entire data processing and model training pipeline."""
    print("Starting full pipeline...")
    run_step(['python', 'src/data_loader.py'])
    run_step(['python', 'src/feature_engineering.py'])
    # This will use the default paths in train.py
    run_step(['python', 'src/train.py'])
    # This will use the default paths in evaluate.py
    run_step(['python', 'src/evaluate.py'])
    print("\nFull pipeline completed successfully!")

def run_prediction(home_form, away_form):
    """Runs the prediction script with given form values."""
    command = [
        'python', 'src/predict.py',
        '--home_form', str(home_form),
        '--away_form', str(away_form)
    ]
    run_step(command)

def run_evolution_step():
    """
    Runs the full model evolution pipeline: retrains a candidate model,
    evaluates it against the production model, and promotes it if it's better.
    """
    print("Starting model evolution step...")

    # Define paths
    prod_model_path = "models/model.joblib"
    prod_encoder_path = "models/label_encoder.joblib"
    prod_metrics_path = "models/metrics.json"

    candidate_model_path = "models/candidate_model.joblib"
    candidate_encoder_path = "models/candidate_encoder.joblib"
    candidate_metrics_path = "models/candidate_metrics.json"

    # 1. Run the data and feature pipeline
    run_step(['python', 'src/data_loader.py'])
    run_step(['python', 'src/feature_engineering.py'])

    # 2. Train a new candidate model
    run_step(['python', 'src/train.py', '--model-path', candidate_model_path, '--encoder-path', candidate_encoder_path])

    # 3. Evaluate the candidate model
    run_step(['python', 'src/evaluate.py',
              '--model-path', candidate_model_path,
              '--encoder-path', candidate_encoder_path,
              '--metrics-path', candidate_metrics_path])
    with open(candidate_metrics_path) as f:
        candidate_metrics = json.load(f)

    # 4. Compare with the production model
    if not os.path.exists(prod_model_path):
        print("No production model found. Promoting candidate model.")
        os.rename(candidate_model_path, prod_model_path)
        os.rename(candidate_encoder_path, prod_encoder_path)
    else:
        # Evaluate the production model
        run_step(['python', 'src/evaluate.py',
                  '--model-path', prod_model_path,
                  '--encoder-path', prod_encoder_path,
                  '--metrics-path', prod_metrics_path])
        with open(prod_metrics_path) as f:
            prod_metrics = json.load(f)

        print(f"\n--- Comparison ---")
        print(f"Candidate Log-Loss: {candidate_metrics['log_loss']:.4f}")
        print(f"Production Log-Loss: {prod_metrics['log_loss']:.4f}")

        # 5. Promote if candidate is better (lower log-loss)
        if candidate_metrics['log_loss'] < prod_metrics['log_loss']:
            print("Candidate model is better. Promoting to production.")
            os.rename(candidate_model_path, prod_model_path)
            os.rename(candidate_encoder_path, prod_encoder_path)
        else:
            print("Production model is better or equal. Discarding candidate.")
            os.remove(candidate_model_path)
            os.remove(candidate_encoder_path)

    print("\nEvolution step completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Football Prediction ML Pipeline Orchestrator")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    subparsers.add_parser('pipeline', help='Run the full data processing and training pipeline.')

    parser_predict = subparsers.add_parser('predict', help='Make a prediction for a single match.')
    parser_predict.add_argument("--home_form", type=float, required=True, help="Form points of the home team.")
    parser_predict.add_argument("--away_form", type=float, required=True, help="Form points of the away team.")

    subparsers.add_parser('evolve', help='Run the model evolution step.')

    args = parser.parse_args()

    if args.command == 'pipeline':
        run_full_pipeline()
    elif args.command == 'predict':
        run_prediction(args.home_form, args.away_form)
    elif args.command == 'evolve':
        run_evolution_step()

if __name__ == "__main__":
    main()
