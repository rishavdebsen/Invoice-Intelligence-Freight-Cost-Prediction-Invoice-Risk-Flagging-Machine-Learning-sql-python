import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load the scaler fitted during training.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):
    """
    Predict whether new vendor invoices should be flagged.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag (1 = flag for review, 0 = normal)
    """
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)
    scaled = scaler.transform(input_df[FEATURES])
    input_df["Predicted_Flag"] = model.predict(scaled)
    return input_df


if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [50, 100, 20],
        "invoice_dollars": [1000, 2500, 400],
        "Freight": [20, 35, 8],
        "total_item_quantity": [50, 100, 18],
        "total_item_dollars": [1000, 2300, 400],
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)