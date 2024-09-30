import pandas as pd
import os
import logging

class DataPreprocessor:
    def __init__(self, file_path, log_dir='logs'):
        self.file_path = file_path
        self.log_dir = log_dir
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging to a file."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        logging.basicConfig(
            filename=os.path.join(self.log_dir, 'preprocessing.log'),
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
        logging.info('Logging setup complete.')

    def load_data(self):
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(self.file_path)
            logging.info(f'Data loaded from {self.file_path}')
            return df
        except Exception as e:
            logging.error(f'Error loading data: {e}')
            raise

    def clean_data(self, df):
        """Clean the data by dropping nulls and converting types."""
        try:
            df.dropna(subset=['Task_Name', 'Duration_Hours', 'Cost_USD'], inplace=True)
            df['Start_Date'] = pd.to_datetime(df['Start_Date'])
            df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')
            df['Error_Rate'] = df['Error_Rate'].fillna(0)
            logging.info('Data cleaned successfully.')
            return df
        except Exception as e:
            logging.error(f'Error cleaning data: {e}')
            raise

    def preprocess(self):
        """Perform end-to-end preprocessing."""
        try:
            df = self.load_data()
            cleaned_df = self.clean_data(df)
            logging.info('Preprocessing completed successfully.')
            return cleaned_df
        except Exception as e:
            logging.error(f'Error during preprocessing: {e}')
            raise

# Usage Example
"""if __name__ == "__main__":
    file_path = 'D:/AI-Enhanced Business Process Optimization/data/business_process_data.csv'
    preprocessor = DataPreprocessor(file_path)
    
    try:
        processed_data = preprocessor.preprocess()
        print(processed_data.head())
        processed_data.to_csv('D:/AI-Enhanced Business Process Optimization/data/cleaned_business_process_data.csv')
    except Exception as e:
        print(f'An error occurred: {e}')"""
