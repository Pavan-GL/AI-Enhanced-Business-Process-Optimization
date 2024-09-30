import os
import logging
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from llm import RecommendationGenerator  # Ensure this imports the correct function

class CostPredictionAPI:
    def __init__(self, model_file, log_dir='logs'):
        self.app = Flask(__name__)
        self.setup_logging(log_dir)
        self.column_names = None
        
        # Load the trained model from a pickle file
        self.model = self.load_model(model_file)
        self.column_names = self.get_column_names(model_file)  # Get column names separately

        # Set up the API route
        self.app.add_url_rule('/predict', view_func=self.predict, methods=['POST'])

    def setup_logging(self, log_dir):
        """Set up logging to a file."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            filename=os.path.join(log_dir, 'api.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        logging.info('Logging setup complete.')

    def load_model(self, model_file):
        """Load the trained model from a pickle file."""
        try:
            with open(model_file, 'rb') as file:
                model_data = pickle.load(file)  # Load the dictionary
                model = model_data['model']  # Extract the model
            logging.info('Model loaded successfully from {}'.format(model_file))
            return model
        except Exception as e:
            logging.error(f'Error loading model: {e}')
            raise

    def get_column_names(self, model_file):
        """Extract column names from the model file."""
        try:
            with open(model_file, 'rb') as file:
                model_data = pickle.load(file)  # Load the dictionary
                column_names = model_data['column_names']  # Extract column names
            logging.info('Column names loaded successfully.')
            return column_names
        except Exception as e:
            logging.error(f'Error loading column names: {e}')
            raise

    def predict(self):
        """Predict cost and generate recommendations based on input data."""
        try:
            task_name = request.json.get('task_name')
            duration_hours = request.json.get('duration_hours')
            error_rate = request.json.get('error_rate')

            # Prepare input for prediction
            input_data = pd.DataFrame([[duration_hours, error_rate, task_name]], 
                                       columns=['Duration_Hours', 'Error_Rate', 'Task_Name'])

            # One-hot encode Task_Name
            input_data = pd.get_dummies(input_data, columns=['Task_Name'], drop_first=True)

            # Ensure all expected columns are present
            for col in self.column_names:
                if col not in input_data.columns:
                    input_data[col] = 0  # Add missing columns with value 0

            # Ensure the input data is ordered the same way as the model was trained
            input_data = input_data[self.column_names]

            logging.info(f'Input Data for Prediction: {input_data}')
            predicted_cost = self.model.predict(input_data)[0]
            logging.info(f'Predicted Cost: {predicted_cost}')
            
            # Generate recommendations using both task name and predicted cost
            recommender = RecommendationGenerator(model_file='D:/AI-Enhanced Business Process Optimization/src/trained_model.pkl')  # Update with your model path
            recommendations = recommender.generate_recommendation(task_name=task_name, predicted_cost=predicted_cost)
            
            logging.info(f'Prediction made for task: {task_name}, Cost: {predicted_cost}, Recommendations: {recommendations}')
            return jsonify({
                'predicted_cost': predicted_cost,
                'recommendations': recommendations
            }), 200
            
        except Exception as e:
            logging.error(f'Error during prediction: {e}')
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model_file_path = 'D:/AI-Enhanced Business Process Optimization/src/trained_model.pkl'  # Update with your model path
    api = CostPredictionAPI(model_file_path)
    api.app.run(debug=True)
