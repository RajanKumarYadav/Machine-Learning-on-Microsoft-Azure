# Capstone Project - Udacity Machine Learning Engineer Nanodegree

This project is about opportunity to use the knowledge we have obtained from this Nanodegree to solve an interesting problem. In this project, we will create two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. Then we will compare the performance of both the models and deploy the best performing model.

We will use an external dataset in our workspace to train a model using the different tools available in the AzureML framework as well we will deploy the model as a web service.

## Project Set Up and Installation

To setup this project we need Azure Cloud Subscription with Admin privilege. Then we have to setup Compute Instance and Compute Cluster, Compute Instance can be used for Jypyter Notebook (Python Programming) and Compute Cluster can be used for run the experiments and deploy the best model on cloud system.After that we will be using both the hyperdrive and automl API from azureml to build this project. We can choose the model we want to train, and the data we want to use. 

## Dataset

In this project we have used Pima Indians Diabetes Database data from Kaggle.com.
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/KaggleDataset.PNG)

Content of the Dataset

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

Dataset Link

https://www.kaggle.com/uciml/pima-indians-diabetes-database


## Project Overview & Workflow

In this project, we have created two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. Then we compared the performance of both the models and deploy the best performing model.

Architecture Diagram

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/capstone-diagram.png)

### Task
In this project we are going to solve Binary Classification Problems in which Target Variable consist Case 0 : Not Diabetic, Case 1 : Diabetic.,Furthur we will create two models: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. Then we will compared the performance of both the models and deploy the best performing model, which can be consumed via any professional API Testing tools.

### Access

We are accessing the data in our workspace by creating the dataset in Microsoft Azure Blob.

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/1.RegisteringDataset.PNG)

## Automated ML

The automl.ipynb file contains code to train a model using Automated ML. 

Run Details :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/2.RunDetails.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/3.RunDetails.PNG)

Experiment Completed :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/4.ExperimentCompleted.PNG)

Choosing Best Model :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/5.BestModel.PNG)

Deploying Best Models :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/6.DeployingBestModels.PNG)

Deploying Best Models :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/7.DeployedBestModel.PNG)

Deployment Using ACI :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/8.DeploymentUsingACI.PNG)

Deployment Using ACI :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/8.DeploymentUsingACI_Success.PNG)

Application Insight Enabled :

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/9.ApplicationInsightEnabled.PNG)

### Results

We have achieved highest Accuracy with VotingEnsemble and Accuracy is 0.78390.

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/5.BestModel.PNG)

## Hyperparameter Tuning

The hyperparameter_tuning.ipynb file contains codes to train a model and perform hyperparameter tuning using HyperDrive.
I have used ML models Scikit-learn to implement.

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/10.hdRunDetails.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/11.hdLog.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/12.hdExperimentsRun.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/13.hdExperimentsCompleted.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/14.BothEndPoints.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/15.hdModelDeployed.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/16.hdAppInsightEnabled.PNG)

![alt text](https://github.com/RajanKumarYadav/Machine-Learning-on-Microsoft-Azure/blob/main/Screenshots/17.hdChildRunSnaps.PNG)


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording

https://youtu.be/jfapMS-l03g

## Standout Suggestions

Converting registered model to ONNX format once deployment is completed
Adjusting features in dataset to address cardinality issue that AutoML had to work through
Implement suggestions above around data imputation, encoding and Handling data imbalance issues

## Acknowledgements

Dataset - Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

Project Concept - Udacity Machine Learning Engineer Nanodegree Program
