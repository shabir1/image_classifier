from subprocess import call
import json
#!flask/bin/python
from flask import Flask, request
from datetime import datetime
from base.tensor_flow_cnn_infer import InferCnn

inf = None
app = Flask(__name__)

@app.route('/ping')
def index():
    return "Pong!"

@app.route('/scan', methods=['POST'])
def scan():
    print (request)
    tt = datetime.now()
    file_in ="{}_{}.{}.{}.{}.{}".format("temp_data/x_in", tt.hour, tt.minute, tt.second, tt.microsecond, "jpg")
    with open(file_in, 'wb') as wfp:
        wfp.write(request.files['image_file'].read())
    result = run_image_classifier_inference(file_in)
    return result

def run_image_classifier_inference(file_in):
    res = inf.infer(file_in)
    return json.dumps(res)

def main():
    call(["ls", "-l"])
    model_directory = '/home/shabir/p-work/PycharmProjects/sk-learn-projects/image_classifier/model_dir/'
    classes = [line.replace("\n", "") for line in open(model_directory + "classes.txt")]
    meta_name = 'model.meta'
    global inf
    inf = InferCnn(classes, model_directory, meta_name)
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    main()
