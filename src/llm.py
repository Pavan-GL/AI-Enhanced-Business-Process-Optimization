import os
import logging
import pickle
from transformers import pipeline

class RecommendationGenerator:
    def __init__(self, model_file, model_name='gpt2', log_dir='logs'):
        self.model_name = model_name
        self.log_dir = log_dir
        self.setup_logging()  # Corrected: no arguments passed
        self.model_file_path = model_file
        
        # Load your recommendation model
        self.model = self.load_model(model_file)

        # Initialize the transformer pipeline
        self.nlp = pipeline('text-generation', model=self.model_name)

    def setup_logging(self):
        """Set up logging to a file."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'recommendation.log'),
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
                column_names = model_data['column_names']  # Store column names
            logging.info('Model loaded successfully from {}'.format(model_file))
            return model  # Return just the model
        except Exception as e:
            logging.error(f'Error loading model: {e}')
            raise

    def generate_recommendation(self, task_name, predicted_cost):
        """Generate recommendations based on task name and predicted cost."""
        try:
            prompt = f"For the task '{task_name}' with a predicted cost of {predicted_cost}, provide 3 concise and actionable optimizations:"
            recommendations = self.nlp(prompt, max_length=200, num_return_sequences=1,truncation=True)
            logging.info(f'Recommendation generated for task: {task_name}')
            return recommendations[0]['generated_text']
        except Exception as e:
            logging.error(f'Error generating recommendation: {e}')
            raise

# Usage Example
if __name__ == "__main__":
    task_name = "Develop new feature"
    predicted_cost = 5000
    model_file_path = 'D:/AI-Enhanced Business Process Optimization/src/trained_model.pkl'  # Update with the correct path

    try:
        recommender = RecommendationGenerator(model_file=model_file_path)
        recommendation = recommender.generate_recommendation(task_name, predicted_cost)
        print(f'Recommendation: {recommendation}')
    except Exception as e:
        print(f'An error occurred: {e}')
