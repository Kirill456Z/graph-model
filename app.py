from flask import Flask, request
from graph_model.model import GraphModel

app = Flask(__name__)

model = GraphModel()


@app.route("/", methods = ['GET'])
def hello_world():
    msg = request.args.get('msg', default = "")
    return model([msg])