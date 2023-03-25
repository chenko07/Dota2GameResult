# Plot the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#creating countplot of label from train and test data
sns.countplot(x='label', data=df_train

sns.countplot(x='label', data=df_test)

#creating countplot of label from cluster and game mode
sns.countplot(x='cluster id', data=df_test)
sns.countplot(x='game mode', data=df_test)
