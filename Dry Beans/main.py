import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from Adaline import *
from Perceptron import *
from Evaluation import *

def preprocess_data_na(data):
    columns_to_process = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Roundness']
    for column in columns_to_process:
        if column in data.columns:
            mean = data[column].mean()
            data[column].fillna(mean, inplace=True)

    return data


data = pd.read_excel('Dry_Bean_Dataset.xlsx')
print(data.to_string())

print("Data Exploration")
print(data.describe())
print(data.info())
print('Shape of data: ',data.shape)
print('number of rows:',data.shape[0])
print('number of columns:',data.shape[1])
print(data.duplicated())
print('sum of duplicated data: ',data.duplicated().sum())
print(data.dtypes)
print('unique in Class: \n ',data['Class'].unique())
print(data['Class'].value_counts())
print("Nulls in each column :")
print(data.isnull().sum().sort_values(ascending=False))

preprocess_data_na(data)
print("Nulls in each column :")
print(data.isnull().sum().sort_values(ascending=False))

data_copy = data.copy()
label_encoder = LabelEncoder()
data_copy['Class'] = label_encoder.fit_transform(data['Class'])

correlation_matrix = data_copy.corr()
print('correlation_matrix :')
print(correlation_matrix)


#plot heatmap
cor = data_copy.corr()
target = abs(cor['Class'])
plt.subplots(figsize=(22, 12))
sns.heatmap(cor, cmap='coolwarm', annot=True, vmin=-1, vmax=1, center=0)
plt.title('Correlation Coefficients')
plt.savefig('heatmap.png')
plt.show()
print("_________________________________________________")


class1 = 'BOMBAY'
class2 = 'SIRA'

df_filtered = data[(data['Class'] == class1) | (data['Class'] == class2)]

X = df_filtered[['MinorAxisLength', 'MajorAxisLength']].values
y = df_filtered['Class'].values

# Encode the class labels
y = y.ravel()
label = LabelEncoder()
y = label.fit_transform(y)
y[y == 0] = -1

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

include_bias = True  # bias

# Augment the feature vectors with a constant value of 1 for the bias term if include_bias is True
if include_bias:
    X = np.c_[X, np.ones(X.shape[0])]

# Split the dataset into train and test sets with a fixed random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=40, stratify=y, random_state=0)

#Adaline Algorithm
adaline = Adaline(learning_rate=0.01, epochs=100)
adaline.train(X_train, y_train)
y_pred_adaline = adaline.predict(X_test)
# Calculate accuracy
print("Adaline Algorithm Evaluation")
accuracy = accuracy_score(y_test, y_pred_adaline)
print(f"Accuracy: {accuracy*100.0:.2f}%")

#confusion_matrix
conf_matrix_adaline = confusion_matrix(y_test, y_pred_adaline)
print("Confusion Matrix:")
print(conf_matrix_adaline)

#overall_accuracy
overall_accuracy = calculate_overall_accuracy(conf_matrix_adaline)
print(f"Overall Accuracy: {overall_accuracy*100.0:.2f}")

#Plot the decision boundary and scatter plot
plot_decision_boundary(adaline, X_test, y_test, class1, class2, include_bias=True)

print("_________________________________________________")

# Perceptron Algorithm
perceptorn = Perceptron(learning_rate=0.01, n_iterations=100)
perceptorn.train_perceptron(X_train, y_train)
y_pred_perceptorn = perceptorn.predict(X_test)
print("Perceptron Algorithm Evaluation")
accuracy = accuracy_score(y_test, y_pred_perceptorn)
print(f"Accuracy: {accuracy*100.0:.2f}%")

#confusion_matrix
conf_matrix_perceptorn= confusion_matrix(y_test, y_pred_perceptorn)
print("Confusion Matrix:")
print(conf_matrix_perceptorn)

#overall_accuracy
overall_accuracy = calculate_overall_accuracy(conf_matrix_perceptorn)
print(f"Overall Accuracy: {overall_accuracy*100.0:.2f}")

#Plot the decision boundary and scatter plot
plot_decision_boundary(perceptorn, X_test, y_test, class1, class2, include_bias=True)

print("_________________________________________________")
