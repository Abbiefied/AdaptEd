from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    student_data = request.json.get('student_data')
    # Implement your recommendation logic here
    recommendations = ["Course 1", "Course 2", "Course 3"]
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
