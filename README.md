# Udacity Machine Learning Engineer with Microsoft Azure Capstone Project

## Table of Contents
* [Overview](#Overview)

## Overview

This is the capstone project for the Udacity Machine Learning Engineer with Microsoft Azure.

In this project I analize the Mexican Government's data for the COVID-19 pandemic from January of 2020 up until the 30th of April of 2021. From this data I intend to get a predictive model to analyze a person's probability to enter an Intensive Care Unit (ICU) based on their COVID-19 test result type, age, gender and comorbidities.

This is the project workflow that was followed.
![Capstone project diagram](images/capstone-diagram.png)

1. Choose a dataset: The dataset chosen for the project is the Mexican Government's General Directorate of Epidemiology COVID-19 Open Data. Available to download in the following URL: https://www.gob.mx/salud/documentos/datos-abiertos-152127
2. Import Dataset into workspace: The dataset in CSV format is registered in the Datasets tab in Azure ML Studio to be used for training.
3. Train model using Automated ML: Using the AzureML SDK for Python a Jupyter Notebook is created where a classification model using AutoML is trained, and the best one is selected.
4. Train model using HyperDrive: Using the AzureML SDK for Python a Jupyter Notebook is created where a classification model using HyperDrive for hyperparameter optimization is trained.
5. Compare model performance: The model with the best accuracy is selected for deployment.
6. Deploy best model: The best model is deployed using Azure Container Instances, a functional endpoint is produced and logging is enabled with Application Insights.
7. Test model endpoint: The endpoint is tested with Apache Benchmark.



## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

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
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

## References
- [Mexican Government's General Directorate of Epidemiology COVID-19 Open Data](https://www.gob.mx/salud/documentos/datos-abiertos-152127)
