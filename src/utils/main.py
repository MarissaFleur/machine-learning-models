import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    # Load the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    data = np.loadtxt(dataset_path, delimiter=',')

    # Split the dataset into features (X) and target (y)
    X = data[:, :-1]
    y = data[:, -1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)