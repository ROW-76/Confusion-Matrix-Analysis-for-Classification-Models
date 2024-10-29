#importing the necessary library
import numpy as np

# Defining the Confusion matrices for Model M1 and Model M2
M1_confusion = np.array([[100, 50], [150, 200]])
M2_confusion = np.array([[200, 90], [10, 200]])

# Defining the Cost matrix
cost_matrix = np.array([[-1, 50], [1, 0]])

# Defining the Function to Calculate_Accuracy for M1 and M2
def calculate_accuracy(confusion_matrix):
    correct_classification = confusion_matrix[0, 0] + confusion_matrix[1, 1]  #Summing up True Negative and True Positive
    total_predictions = confusion_matrix.sum()  # Total number of predictions (TP/ TN/ FP/ FN)
    return correct_classification / total_predictions

# Defining the function to Calculate_Cost
def calculate_cost(confusion_matrix, cost):
    return np.sum(confusion_matrix * cost)

# Calculating the Accuracy for Model M1
M1_Accuracy = calculate_accuracy(M1_confusion)
M2_Accuracy = calculate_accuracy(M2_confusion)

# Calculating the Cost for Model M2
M1_Cost = calculate_cost(M1_confusion, cost_matrix)
M2_Cost = calculate_cost(M2_confusion, cost_matrix)

# Displaying the results
print(f"Model M1: Accuracy = {M1_Accuracy * 100}%, Cost = {M1_Cost}")
print(f"Model M2: Accuracy = {M2_Accuracy * 100}%, Cost = {M2_Cost}")