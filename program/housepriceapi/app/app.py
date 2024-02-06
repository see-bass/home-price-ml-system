#import base64
import json
#import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model as md
#############

# from PIL import Image
# from io import BytesIO
from pathlib import Path

# New
import tarfile
import tempfile

from flask import Flask

import os
import joblib
import sklearn
import json
import pandas as pd
from io import StringIO

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


#model_file = '/opt/ml/model'
#model = tf.keras.models.load_model(model_file)

MODEL_PATH = Path(__file__).parent

class Model:
    model = None

    feat = None
    targ = None

    TARGET_COLUMN = "price"
    FEATURE_COLUMNS = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "waterfront",
        "view",
        "grade",
        "sqft_living15",
]

    def load(self):
        """
        Extracts the model package and loads the model in memory if it hasn't
        been loaded yet.
        """

        # We want to load the model only if it is not loaded yet.
        if not Model.model:
            # Before we load the model, we need to extract it in a
            # temporal directory.
            with tempfile.TemporaryDirectory() as directory:
                with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
                    tar.extractall(path=directory)

                #Model.model = tf.keras.models.load_model(directory)
                #Model.model = layers.TFSMLayer(directory,call_endpoint="serving_default")

                tfsm_layer = layers.TFSMLayer(directory,call_endpoint="serving_default")
                inputs = layers.Input(shape=(7,))
                outputs = tfsm_layer(inputs)
                Model.model = md(inputs=inputs, outputs=outputs)

                Model.feat = joblib.load(os.path.join(directory, "features.joblib"))

                targ_t = joblib.load(os.path.join(directory, "target.joblib"))
                Model.targ = targ_t.named_transformers_["price"]

    def ensure_json_format(self, data):
    # Check if data is a dictionary
        if isinstance(data, dict):
            return json.dumps(data)  # Convert dict to JSON-formatted string
        
        # Check if data is a string
        elif isinstance(data, str):
            try:
                json.loads(data)  # Try parsing the string as JSON
                return data  # It's already a JSON-formatted string
            except json.JSONDecodeError:
                # Handle the case where the string is not JSON-formatted
                # You can decide how to handle this (raise an error, return None, etc.)
                raise ValueError("Provided string is not a valid JSON-formatted string")        
    
    def input_fn(self, input_data, content_type="application/json"):
        """
        Parses the input payload and creates a Pandas DataFrame.
        """

        input_data = self.ensure_json_format(input_data)

        if content_type == "array":
            df = pd.DataFrame(input_data, columns=Model.FEATURE_COLUMNS)

            return df

        if content_type == "text/csv":
            df = pd.read_csv(StringIO(input_data), header=None, skipinitialspace=True)

            df.columns = Model.FEATURE_COLUMNS
            return df

        if content_type == "application/json":
            df = pd.DataFrame([json.loads(input_data)])

            return df

        raise ValueError(f"{content_type} is not supported!")

    
    def predict(self, data, content_type="application/json"):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        
        reshape_data = self.input_fn(data,content_type)
        data_scaled = Model.feat.transform(reshape_data)
        pred = Model.model.predict(data_scaled)
        prednum  = pred['dense_2']
        predictions = Model.targ.inverse_transform(prednum)

        return predictions


def generate_response(status_code, body, cors=True):
    """
    Generate a response with the given status code, body, and CORS headers.
    :param status_code: HTTP status code.
    :param body: Response body, which will be converted to JSON.
    :param cors: Boolean to indicate if CORS headers should be included.
    :return: A response dictionary with statusCode, headers, and body.
    """
    response = {
        "statusCode": status_code,
        "body": json.dumps(body)
    }

    if cors:
        response["headers"] = {
            'Access-Control-Allow-Origin': '*',  # or specific domain for production
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': '*'
        }

    return response

app = Flask(__name__)
model = Model()

@app.route("/predict/", methods=["POST"])

def lambda_handler(event, context):
    try:
        logger.info(f"Incoming event: {event}")
        data = event["body"]

        content_type = 'application/json'
        if 'headers' in event and event['headers'] is not None:
        # Get Content-Type header (case-insensitive)
            content_type = event['headers'].get('Content-Type') or event['headers'].get('content-type') or content_type
        
        #content_type = event["headers"]["Content-Type"]
        # image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
        # image = image.resize((28, 28))

        # probabilities = model(np.array(image).reshape(-1, 28, 28, 1))
        # label = np.argmax(probabilities)

        predictions = model.predict(data=data,content_type=content_type)    
        prediction = float(predictions[0])

        #return jsonify({"prediction": prediction})

        return generate_response(200, {"prediction": prediction})

    # return {
    #     'statusCode': 200,
    #     'headers': {
    #         'Access-Control-Allow-Origin': '*',  # or specific domain for production
    #         'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
    #         'Access-Control-Allow-Methods': '*'
    #     },
    #     'body': json.dumps(
    #         {
    #             "prediction": prediction,
    #         }
    #     )
    # }

    except Exception as e:
        # Handle exceptions and return an error response
        return generate_response(500, {"error": str(e)})