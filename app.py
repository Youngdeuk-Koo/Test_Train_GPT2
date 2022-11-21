from flask import Flask, request, jsonifyDataset
from flask_api import FlaskAPI, status
import chatbot
import chatbot_test2
import json

# config = {
#     "DEBUG": True,
#     "CACHE_TYPE": "simple",
#     "CACHE_DEFAULT_TIMEOUT": 300
# }

app = FlaskAPI(__name__)
# app.config.from_mapping(config)
# cache = Cache(app)

# cache = Cache(config={'CACHE_TYPE': 'simple'})

# app = Flask(__name__)
# cache.init_app(app)

# cache = Cache(config={'CACHE_TYPE': 'simple'})
# cache.init_app(app, config={'CACHE_TYPE': 'simple'})


# @app.route('/', methods=["GET", "POST"])
# def chatbot_response():
#     if request.method == 'POST':
#         the_question = request.data
#         _data = json.loads(the_question)
#         # response = chatbot_response(_data)
#         response = chatbot_test2.test(_data)
        
#     return jsonify({"answer":response})

@app.route("/api/response", methods=['GET', 'POST'])
def api():

    # chat message
    if request.method == 'POST':
        
        _data = request.data
        if isinstance(_data, str):
            _data = json.loads(_data)

        response = chatbot.test(_data)
        return json.dumps(response, ensure_ascii=False), status.HTTP_200_OK

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='11111', debug=True)