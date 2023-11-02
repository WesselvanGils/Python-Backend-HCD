import os
import json
import numpy as np

from model import Model
from flask_cors import CORS
from flask import Flask, request, send_file

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

model = Model()

@app.route("/api/divorce", methods=["POST"])
def handle_post_request():
    # Convert the data to a numpy array
    post_data = request.get_data()
    post_data_str = post_data.decode("utf-8")
    json_data = json.loads(post_data_str)
    values_array = list(json_data.values())
    numpy_array = np.array(values_array, dtype=np.int32)

    # Make the prediction and generate the image
    pred, conf = model.predict(numpy_array)

    # Generate the JSON response
    response_dict = {"prediction": str(pred), "confidence": str(conf)}
    json_string = json.dumps(response_dict)
    return json_string, 200

@app.route("/image", methods=["GET"])
def handle_image_request():
    return send_file("./temp.png", mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
