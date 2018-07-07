from flask import Flask, json, request

from flask_restplus import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0.0', title='Python Run Environment API for Datoin',
          description='A simple RESTfull API to run python code from Datoin Modules',
          )

# install dependencies on a fresh virtualenv like
# pip install -r requirements.txt

ns = api.namespace('DPRE', description='Datoin Python Runtime Environment')

result = api.model('Result', {
    'result': fields.String(required=True, description='The task results'),
    'time': fields.Float(required=True, description='The task runtime')
})

request1 = api.model('Request', {
    'input_url': fields.String(required=True, description='The url of the image')})

parserT = api.parser()
parserT.add_argument('body', type=dict, required=True, help='The datoin document', location='json')


@ns.route('/dpme')
class Task(Resource):
    """Shows results and runtime of the task submited"""

    @api.doc(parser=parserT)
    #@api.marshal_with(result, code=201)
    def post(self):
        """ run a task """
        all_params = request.json
        # return {"result1": "self.dpr.run(all_params)", "time2":30}, 200
        return IMGCLS.run(all_params), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5522)

