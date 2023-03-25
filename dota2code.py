#----- Project Dota 2 Game Result -----

#Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report

# Load the Dota 2 dataset
df_train = pd.read_csv("dota2Train.csv", delimiter = ";")
df_test = pd.read_csv("dota2Test.csv", delimiter = ";")

# Split the training set into input features and target labels
X_train = df_train.drop("game mode", axis=1)
y_train = df_train["game mode"]
df_train

# Split the test set into input features and target labels
X_test = df_test.drop("game mode", axis=1)
y_test = df_test["game mode"]

# Scale the input features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the MLP model
input_layer_size = 117
output_layer_size = 1

model = MLPClassifier(hidden_layer_sizes=(117,117,117), 
                      activation='relu', 
                      solver='sgd',
                      learning_rate_init=0.001,
                      # alpha = 0.0005,
                      max_iter=100)

# Train the model on the training set
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test_scaled, y_test)
print("Test set accuracy: {:.2f}".format(accuracy)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled) 
      
# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
      
# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# evaluasi model
scores = model.score(X_train_scaled, y_train)
print("Accuracy: %.2f%%" % (scores*100))
      
      
      
