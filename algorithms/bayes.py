import pandas as pd
import numpy as np

def bayes_algorithm(data, feature_list=None, feature_values=None, laplace_smoothing=False):
    # Input validation
    if not feature_list or not feature_values:
        return {
            'error': "Insufficient information for prediction",
            'prediction': None,
            'dataframe': None,
            'html_table': None
        }
    
    # Check feature list and values match
    if len(feature_list) != len(feature_values):
        return {
            'error': "Number of features and values do not match",
            'prediction': None,
            'dataframe': None,
            'html_table': None
        }
    
    # Identify target column (last column)
    target_column = data.columns[-1]
    
    # Filter data based on input features
    filtered_data = data.copy()
    for feature, value in zip(feature_list, feature_values):
        # if isinstance(filtered_data[feature][0], str) == True:
        filtered_data = filtered_data[filtered_data[feature] == value]
        # else:
        #     filtered_data = filtered_data[filtered_data[feature] == float(value)]
    
    # Check if any data remains after filtering
    if len(filtered_data) == 0:
        return {
            'error': "No data found matching the specified features",
            'prediction': None,
            'dataframe': None,
            'html_table': None
        }
    
    # Get unique classes
    classes = filtered_data[target_column].unique()
    
    # Calculate class probabilities
    class_probabilities = {}
    total_samples = len(filtered_data)
    
    for cls in classes:
        # Prior probability of the class
        class_count = len(filtered_data[filtered_data[target_column] == cls])
        class_probabilities[cls] = class_count / total_samples
    
    # Apply Laplace smoothing if requested
    if laplace_smoothing:
        total_classes = len(classes)
        class_probabilities = {
            cls: (count + 1) / (total_samples + total_classes)
            for cls, count in class_probabilities.items()
        }
    
    # Select class with highest probability
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    
    # Prepare result data for formatting
    result_data = [
        {
            'Đặc trưng': feature,
            'Giá trị': value
        } for feature, value in zip(feature_list, feature_values)
    ]
    
    # Add prediction to result data
    result_data.append({
        'Đặc trưng': 'Kết quả dự đoán',
        'Giá trị': str(predicted_class)
    })
    
    # Create DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Create HTML table
    html_table = result_df.to_html(
        classes='table table-striped table-bordered', 
        index=False, 
        escape=False
    )
    return html_table