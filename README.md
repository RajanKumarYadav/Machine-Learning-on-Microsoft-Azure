# Capstone Project - Udacity Machine Learning Engineer Nanodegree

This project is about opportunity to use the knowledge we have obtained from this Nanodegree to solve an interesting problem. In this project, we will create two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. Then we will compare the performance of both the models and deploy the best performing model.

We will use an external dataset in our workspace to train a model using the different tools available in the AzureML framework as well we will deploy the model as a web service.

## Project Set Up and Installation

To setup this project we need Azure Cloud Subscription with Admin privilege. Then we have to setup Compute Instance and Compute Cluster, Compute Instance can be used for Jypyter Notebook (Python Programming) and Compute Cluster can be used for run the experiments and deploy the best model on cloud system.After that we will be using both the hyperdrive and automl API from azureml to build this project. We can choose the model we want to train, and the data we want to use. 

## Dataset

In this project we have used Pima Indians Diabetes Database data from Kaggle.com.
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/KaggleDataset.PNG)

Content

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

Dataset Link

https://www.kaggle.com/uciml/pima-indians-diabetes-database





### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

https://youtu.be/jfapMS-l03g

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## Acknowledgements

Dataset - Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

Project Concept - Udacity Machine Learning Engineer Nanodegree Program
