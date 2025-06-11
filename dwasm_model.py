import numpy as np
from sklearn.metrics import accuracy_score

def predict_dwasm(models, weights, X):
    """
    Predict loan approval using DWASM ensemble logic.
    Args:
        models: list of trained models [xgb, gb, rf]
        weights: softmax weights for the models
        X: scaled and preprocessed input features
    Returns:
        Array of predictions (0 or 1)
    """
    # Safety check to catch incorrect input
    for i, model in enumerate(models):
        if not hasattr(model, "predict_proba"):
            raise TypeError(f"Item at index {i} is not a model with predict_proba(). Got: {type(model)}")

    probabilities = [model.predict_proba(X)[:, 1] for model in models]
    weighted_sum = sum(w * p for w, p in zip(weights, probabilities))
    return (weighted_sum >= 0.5).astype(int)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model on test data.
    Args:
        models: list of models [xgb, gb, rf]
        X_test: test features
        y_test: test labels
    Returns:
        Dictionary of model names and their accuracies
    """
    names = ["XGBoost", "Gradient Boosting", "Random Forest"]
    return {
        name: accuracy_score(y_test, model.predict(X_test))
        for name, model in zip(names, models)
    }
