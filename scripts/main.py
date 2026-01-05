import logging
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from machine_learning_models.data_loader import load_data
from machine_learning_models.data_preprocessor import preprocess_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a random forest classifier')
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    return parser.parse_args()

def train_model(X_train, y_train, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    args = parse_arguments()
    data = load_data(args.data_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    model = train_model(X_train, y_train, args.n_estimators)
    accuracy, report = evaluate_model(model, X_test, y_test)
    logging.info(f'Model accuracy: {accuracy:.3f}')
    logging.info(report)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()