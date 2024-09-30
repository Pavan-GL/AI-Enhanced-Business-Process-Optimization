import pickle

model_file =  'D:/AI-Enhanced Business Process Optimization/src/trained_model.pkl'

try:
    with open(model_file, 'rb') as file:
        model_data = pickle.load(file)
        print(type(model_data))  # Should be a dict
        print(model_data.keys())  # Should include 'model' and 'column_names'
except Exception as e:
    print(f'Error loading model data: {e}')
