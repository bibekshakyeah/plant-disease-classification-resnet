from flask import Flask, jsonify, request
from CNNRESNET import predict_image,model

# Define Flask app
app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def index():
    print('here')

    return 'Hello, World!'

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print('here')

    # Get image data from request
    img_bytes = request.get_data()
    
    # Make prediction and return result as JSON
    result = {'prediction': predict_image(img_bytes,model)}
    return jsonify(result)

# Run app
if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0',port='5001')

