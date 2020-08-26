from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from nltk.corpus import stopwords
import string
import torch
import torch.nn as nn
import numpy as np
import csv
import sys
import os
from char_cnn import CharCNN


csv.field_size_limit(sys.maxsize)


#-------------------------------------flask backend-----------------------------------------------

flask_app = Flask(__name__, static_folder="build", static_url_path="")

# @flask_app.route('/', methods=["GET"])
# def index():
#     return flask_app.send_static_file('index.html')
#
# @flask_app.route('/favicon.ico', methods=["GET"])
# def favicon():
#     return flask_app.send_static_file('favicon.ico')
#
#
# @flask_app.route('/prediction', methods=["GET","POST"])
# def make_prediction():
#     print("in pred")
#     try:
#         formData = request.json
#         print(formData)
#         data = [val for val in formData.values()]
#         # print(data)
#         prediction = predict_gender(data)
#         # prediction = predict.prediction(data)
#
#         response = jsonify({
#             "statusCode": 200,
#             "status": "Prediction made",
#             "result": "This text was written by a : " + str(prediction)
#             })
#         # print(response)
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response
#     except Exception as error:
#         return jsonify({
#             "statusCode": 500,
#             "status": "Could not make prediction",
#             "error": str(error)
#         })


app = Api(app = flask_app,
		  version = "1.0",
		  title = "Gender predictor app",
		  description = "Predict gender using a trained model")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params',
                    {'text': fields.String(required=True,
                    description="User text",
                    help="User text field cannot be empty")})


@name_space.route("/")
class MainClass(Resource):

    def options(self):

        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        try:
            formData = request.json
            # print(formData)
            data = [val for val in formData.values()]
            # print(data)
            prediction = predict_gender(data)
            # prediction = predict.prediction(data)

            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": "This text was written by a : " + str(prediction)
                })
            # print(response)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })



#----------------------------------------------Inference module------------------------------------------------------


def predict_gender(text):
    res = ""
    words = stopwords.words("english")
    table = str.maketrans('', '', string.punctuation)
    # text = "This thing is not good at all. Do not buy it."
    text = text[0]
    cleaned_text = " ".join([i.translate(table) for i in text.split() if i.isalpha() if i not in words]).lower()
    max_length = 1014
    vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'/\|_@#$%ˆ&*˜‘+-=<>()[]{}""")
    identity_mat = np.identity(len(vocabulary))

    data = np.array([identity_mat[vocabulary.index(i)] for i in list(cleaned_text) if i in vocabulary],
                    dtype=np.float32)

    if len(data) > max_length:
        data = data[:max_length]
    elif 0 < len(data) < max_length:
        data = np.concatenate(
            (data, np.zeros((max_length - len(data), len(vocabulary)), dtype=np.float32)))
    elif len(data) == 0:
        data = np.zeros((max_length, len(vocabulary)), dtype=np.float32)


    model = torch.load("model_amz_ccnn").cpu()
    model.eval()

    data = torch.FloatTensor(data).cpu()
    data = data.unsqueeze(dim=0)
    # print(data.shape)
    prediction = model(data)
    # print(prediction)
    prediction = torch.argmax(prediction)
    if prediction == 0:
        res = "Male"
    else:
        res = "Female"

    return res


#-----------------------------------------------------MAIN------------------------------------------------------


if __name__ == "__main__":

    if (os.environ.get('PORT')):
        port = int(os.environ.get('PORT'))
    else:
        port = 5000
    flask_app.run(host="127.0.0.1", debug=True, port=port)
    # res = predict_gender("This is not a good product. Please avoid it")
    # print(res)