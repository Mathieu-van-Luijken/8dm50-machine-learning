import numpy as np
from sklearn.datasets import load_breast_cancer
from scipy.stats import norm
import pandas as pd

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
class_labels = data.target_names

# Separate the data into two classes: benign (class 0) and malignant (class 1)
X_class_0 = X[y == 0]  # benign
X_class_1 = X[y == 1]  # malignant

# Function to compute class-conditional probabilities assuming a Gaussian distribution
def compute_class_conditional_probabilities(X_class_0, X_class_1, X_input, feature_names):
    # Initialize an empty list to store results
    result_list = []
    
    for i, feature_name in enumerate(feature_names):
        # Get feature values for class 0 and class 1
        feature_values_class_0 = X_class_0[:, i]
        feature_values_class_1 = X_class_1[:, i]
        
        # Calculate mean and std for both classes
        mean_0, std_0 = np.mean(feature_values_class_0), np.std(feature_values_class_0)
        mean_1, std_1 = np.mean(feature_values_class_1), np.std(feature_values_class_1)
        
        # Compute the probability P(X = x | Y = class_0) and P(X = x | Y = class_1) using Gaussian PDF
        prob_class_0 = norm.pdf(X_input[i], mean_0, std_0)
        prob_class_1 = norm.pdf(X_input[i], mean_1, std_1)
        
        # Append the result as a dictionary
        result_list.append({
            'Feature': feature_name,
            'Class_0_Prob': prob_class_0,
            'Class_1_Prob': prob_class_1
        })
    
    # Convert the list of dictionaries to a DataFrame
    class_conditional_probs = pd.DataFrame(result_list)
    
    return class_conditional_probs

# Example input X (you can replace this with any sample from the dataset)
X_input = X[0]  # Taking the first sample as an example

# Compute class-conditional probabilities for the input sample
class_conditional_probs = compute_class_conditional_probabilities(X_class_0, X_class_1, X_input, feature_names)

# Display the class-conditional probabilities
print(class_conditional_probs)
