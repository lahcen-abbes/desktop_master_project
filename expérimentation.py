import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def svm_classification(data_path, iterations=100):
    """
    Train SVM classifier with a linear kernel and calculate Sensitivity and Specificity scores.
    
    Args:
    - data_path (str): Path to the CSV file containing the dataset.
    - iterations (int): Number of iterations to execute the code.
    
    Returns:
    - None
    """
    # Initialize variables to store Sensitivity and Specificity scores
    sensitivity_scores = []
    specificity_scores = []
    
    # Execute the code for the specified number of iterations
    for _ in range(iterations):
        # Read the CSV file
        df = pd.read_csv(data_path, sep=";")
        
        # Check if the 'diagnosis' column exists in the DataFrame
        if 'diagnosis' in df.columns:
            # Randomly split data into training and testing sets
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Calculate the number of patients in the test set (lines)
            patients_test_set = len(y_test)
            
            # Data preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train the SVM classifier with the linear kernel
            svm_classifier = SVC(kernel='linear')
            svm_classifier.fit(X_train_scaled, y_train)
            
            # Make predictions on the test set
            y_pred = svm_classifier.predict(X_test_scaled)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calculate Sensitivity (True Positive Rate)
            sensitivity = tp / (tp + fn)
            sensitivity_scores.append(sensitivity)
            
            # Calculate Specificity (True Negative Rate)
            specificity = tn / (tn + fp)
            specificity_scores.append(specificity)
        else:
            print("The 'diagnosis' column does not exist in the DataFrame.")
    
    print("Number of patients (lines) used for calculating Sensitivity and Specificity:", patients_test_set)  # Print the number of patients (lines) in the test set for reference
    # Print the average Sensitivity and Specificity scores
    print("True Negatif:", tn)
    print("False Positif:", fp)
    print("False Negatif:", fn)
    print("True Positif:", tp)
    print("Average Sensitivity:", sum(sensitivity_scores) / iterations)
    print("Average Specificity:", sum(specificity_scores) / iterations)
svm_classification("C:/Users/LAHCENE/OneDrive/Bureau/tkinter project/data.csv", iterations=100)
