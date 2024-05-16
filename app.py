"""
This module defines the Flask application for the game design recommendation service.

The application provides an API endpoint that receives player data in JSON format,
validates the data, generates predictions using the recommender system, and returns
the recommendations as structured JSON responses. The module handles all aspects
of request processing, including data validation, error handling, and response formatting.
"""

import logging
import uuid
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, jsonify, request

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
    """
    Process a POST request containing player data in JSON format and return design recommendations.

    The function expects a JSON payload with player data, validates the format and required fields,
    then uses a recommender system to generate design recommendations for each player. The results are
    returned as a JSON response structured to include the recommendations along with metadata such as
    request ID and timestamp.

    If the incoming JSON is improperly formatted or if an error occurs during processing, the function
    responds with an appropriate error message and status code.

    Returns:
        A Flask JSON response containing either the recommendations in a structured format or
        an error message with a corresponding HTTP status code.
    """
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
