from flask import Flask, request
from graph_model.model import GraphModel
import json

app = Flask(__name__)

model = GraphModel()

FINAL_NODE = 6

@app.route("/", methods = ['GET'])
def hello_world():
    msg = request.args.get('msg', default = "")
    response, last_node = model([msg])
    return json.dumps({"response" : response, "is_terminated" : (last_node == FINAL_NODE)})