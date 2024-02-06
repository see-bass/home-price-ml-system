import tarfile
import tempfile
import numpy as np

from flask import Flask, request, jsonify
from pathlib import Path
from tensorflow import keras
#from keras import layers

import os
import joblib
import sklearn
import json
import pandas as pd
from io import StringIO


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

                Model.model = keras.models.load_model(directory)
                #Model.model = layers.TFSMLayer(directory,call_endpoint="serving_default")

                Model.feat = joblib.load(os.path.join(directory, "features.joblib"))

                targ_t = joblib.load(os.path.join(directory, "target.joblib"))
                Model.targ = targ_t.named_transformers_["price"]

    def input_fn(self, input_data, content_type="array"):
        """
        Parses the input payload and creates a Pandas DataFrame.
        """

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
    
    def predict(self, data):
        """
        Generates predictions for the supplied data.
        """
        self.load()
        
        reshape_data = self.input_fn(data,"array")
        data_scaled = Model.feat.transform(reshape_data)
        pred = Model.model.predict(data_scaled)
        predictions = Model.targ.inverse_transform(pred)

        return predictions


app = Flask(__name__)
model = Model()


@app.route("/predict/", methods=["POST"])
def predict():
    data = request.data.decode("utf-8")
    data = np.array(data.split(",")).astype(np.float32)

    predictions = model.predict(data=[data])    
    prediction = float(predictions[0])

    return jsonify({"prediction": prediction})
