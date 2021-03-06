from flask import Flask, request, jsonify
from flask_restx import Resource, Api, fields, reqparse, abort
from run import Swapping
from waitress import serve
app = Flask(__name__)

# @app.route('/predict',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     source_path,target_path = data['Source'],data['Target']
#     predicted_path = Swapping.predict(source_path, target_path)

#     return jsonify(output)
api = Api(app, version="1.0", title="FSGAN", description="Input profile and target video")
app.config['SWAGGER_UI_JSONEDITOR']=True
ns = api.namespace('Names', description='Program is trained to do FaceSwapping.')
Swapping_model=Swapping()
parse = reqparse.RequestParser()
parse.add_argument("Input Path", required=True, type=str)
parse.add_argument("Target Path", required=True, type=str)

feat = {}
# def load_model():
#     if Swapping==None:
#         Swapping=Swapping()

def abort_req(id):
    if id not in feat:
        abort(404, message="Invalid ID")

class Test(Resource):
    def get(self,id):
        abort_req(id)
        inputs = feat[id]
        source_path,target_path = inputs["Input Path"],inputs["Target Path"]
        predicted_path = Swapping_model.predict(source_path, target_path)
        return {"Output Path: ":predicted_path}
        
    @api.expect(parse)
    def post(self,id):
        args = parse.parse_args()
        feat[id] = args
        return feat[id], 201
        
    
api.add_resource(Test, "/<int:id>")
if __name__ == '__main__':
    serve(app)
