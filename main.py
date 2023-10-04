import torch
import json
import numpy as np

from model import Net
from flask import Flask, request

app = Flask(__name__)

HIDDEN_SIZE = 100
HIDDEN_COUNT = 2

net = Net(20, HIDDEN_SIZE, HIDDEN_COUNT, 2)
net.load_state_dict(torch.load("./models/model.pth"))


@app.route("/api/divorce", methods=["POST"])
def handle_post_request():
    post_data = request.get_data()
    post_data_str = post_data.decode("utf-8")
    json_data = json.loads(post_data_str)
    values_array = list(json_data.values())
    numpy_array = np.array(values_array)

    tensor = torch.Tensor(numpy_array)

    with torch.no_grad():
        outputs = net(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=0)
        conf, classes = torch.max(probs, 0)
        class_names = '0123456789'
        confidence = conf.item()
        prediction = class_names[classes.item()]

    json_string = '{"prediction":' + prediction + ","
    other = '"confidence":' + f'{confidence * 100:.2f}%' + "}"

    json_string += other

    return f"{json_string}", 200
