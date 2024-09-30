import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class ModelTrainer:
    def __init__(self, data, log_dir='logs',model_dir = 'src'):
        self.data = data
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        self.setup_logging()

    def setup_logging(self):
        """Set up logging to a file."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'model_training.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        logging.info('Logging setup complete.')

    def train_model(self):
        """Train a RandomForest model and evaluate it."""
        try:
            print("Columns in the DataFrame:", self.data.columns.tolist())
            self.data.columns = self.data.columns.str.strip()

            # Prepare features and target
            required_columns = ['Duration_Hours', 'Error_Rate', 'Task_Name']
            
            # Check if required columns are in the DataFrame
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f'Missing columns in the DataFrame: {missing_columns}')
            
            X = self.data[required_columns]
            y = self.data['Cost_USD']
            print("Features (X):", X.head())
            print("Target (y):", y.head())

            # One-hot encode Task_Name
            X = pd.get_dummies(X, columns=['Task_Name'], drop_first=True)
            print(X)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info('Data split into training and testing sets.')

            # Train the model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            logging.info('Model training complete.')

            # Predictions
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info(f'Mean Squared Error: {mse}')

            # Save the model and its column names
            model_file_path = os.path.join(self.model_dir, 'trained_model.pkl')
            model_data = {
                'model': model,
                'column_names': X.columns.tolist()  # Store the column names after encoding
            }
            with open(model_file_path, 'wb') as model_file:
                pickle.dump(model_data, model_file)
               
            logging.info(f'Model saved to {model_file_path}')

            return model, mse
        except Exception as e:
            logging.error(f'Error during model training: {e}')
            raise

# Usage Example
if __name__ == "__main__":
    # Load your data first
    data_file_path = 'D:/AI-Enhanced Business Process Optimization/data/cleaned_business_process_data.csv'
    
    try:
        data = pd.read_csv(data_file_path)  # Ensure the data is preloaded or cleaned as needed
        model_trainer = ModelTrainer(data)
        trained_model, error = model_trainer.train_model()
        print(f'Trained Model: {trained_model}\nMean Squared Error: {error}')
    except Exception as e:
        print(f'An error occurred: {e}')
