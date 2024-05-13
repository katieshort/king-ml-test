from flask import Flask, request, jsonify
from datetime import datetime, timezone
import pandas as pd
import logging
import uuid
from src.models.recommender import GameDesignRecommender
from src.utils.utils import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load configuration
config = load_config("config/config.yaml")
recommender = GameDesignRecommender(config)


def generate_request_id():
    """Generate a unique request ID for each request."""
    return uuid.uuid4().hex


def validate_json(json_data):
    """Validate and convert incoming JSON data to a DataFrame."""
    if not json_data:
        raise ValueError("No data provided")
    if isinstance(json_data, dict):
        json_data = [json_data]
    df = pd.DataFrame(json_data)
    required_columns = {"id"}  # Add other required column names here
    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required data fields")
    if not all(isinstance(item, str) for item in df["id"]):
        raise ValueError("ID must be a string")
    return df


def build_response(prediction_responses):
    """Build a structured response for the API."""
    return {
        "status": "success",
        "predictions": prediction_responses,
        "metadata": {
            "request_id": generate_request_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()
        query_df = validate_json(json_data)

        predictions = recommender.predict(query_df)

        prediction_responses = [
            {"player_id": item["id"], "design_recommendation": pred}
            for item, pred in zip(json_data, predictions)
        ]

        return jsonify(build_response(prediction_responses))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
