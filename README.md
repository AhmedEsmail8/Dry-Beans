# Dry Beans Classification
![beans](https://github.com/Yasmine-Khaled/Dry-Beans/assets/89998528/6a936b99-46a9-4ec2-9f5b-deff7c81391d)

## View the Results
The results of the project, including visualizations and analysis, can be found in the Kaggle Notebook:
https://www.kaggle.com/code/yasmine5aled/dry-beans

## Project Description:

The "Dry Beans Classification" project aims to predict and classify three distinct categories of dry beans: Bombay, Cali, and Sira. This classification is based on measurements of five significant features: Area, Perimeter, MajorAxisLength, MinorAxisLength, and Roundness, each measured in millimeters. The dataset contains a total of 150 samples, with 50 samples from each of the three dry bean varieties.

## Objective:

The primary objective of this project is to build a machine learning model that can accurately classify dry beans into one of the three categories (Bombay, Cali, or Sira) based on the provided measurements. 

## Project Stages:

### 1. Data Exploration and Analysis:

* Exploring the dataset to gain insights into the distribution of features and class labels.

* Visualizing data to understand feature relationships and distributions.

### 2. Data Preprocessing:

* Preparing the data for model training by handling missing values, scaling, and encoding categorical target.

* Splitting the data into training and testing sets (Each class has 50 samples: train NN with 30 non-repeated samples randomly selected, and test it with the remaining 20 samples).

* Feature scaling using StadardScalar().

### 3. Model Implementation:

* Adaline Algorithm: We implemented the Adaline (Adaptive Linear Neuron) algorithm from scratch. Adaline is a single-layer neural network that uses linear activation and minimizes Mean Squared Error (MSE) during training. We coded the algorithm to handle the training process, weight updates, and predictions.

* Perceptron Algorithm: Additionally, we implemented the Perceptron algorithm from scratch. The Perceptron is a simple binary classification algorithm that updates its weights based on misclassified instances. We implemented the Perceptron algorithm to classify dry beans into the provided categories.

### 4. Model Training:

For both Adaline and Perceptron, we trained the models using the training data. We iteratively updated the model parameters to minimize classification errors.

### 5. Model Evaluation:

* We evaluated the performance of both the Adaline and Perceptron models on the test dataset.
* We constructed Confusion matrices and calculated metrics such as Overall Accuracy.

### 6. Graphical User Interface (GUI):

We developed a user-friendly GUI that allows users to interact with the model. 

#### The GUI includes the following features:

* **User Input:** Users can select two features, two classes, specify the learning rate (eta), set the number of epochs (m), enter an MSE threshold, and choose whether to add a bias term.

* **Algorithm Selection:** Users can choose between the Perceptron or Adaline algorithm for classification.

* **Test Data Input:** Users can browse and upload a test file or manually input data for prediction.

* **Prediction:** The GUI provides real-time prediction based on the selected features, classes, and model settings.

* **Overall Accuracy:** After making predictions, the GUI displays the overall accuracy of the model's predictions.

* **Intuitive Controls:** The GUI offers an intuitive interface for users to interact with the model without requiring programming knowledge.

## Screenshots :



**Feel free to reach out with any questions or suggestions!**
