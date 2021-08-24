# Capstone Project: Azure Machine Learning Engineer

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
Explain any special installation steps - explain how to set up this project in AzureML.

## Dataset

### Overview
The dataset we will be using in this project is called *Heart failure clinical records Data Set* and is publicly available from UCI Machine Learning Repository. 

The dataset contains medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Here is a [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv) to the data.  


### Task

The task we are concerned with is to predict if the patient deceased during the follow-up period. We will be using `DEATH_EVENT` column as the target and since this is a boolean variable, the task at hand is Binary Classification.


The 12 clinical input features and the target feature are as follows:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)


### Access
In the cell below, we write code to access the data that we will store and use on Azure:

```python
found = False
key = "Health-Failure"
description_text = "Health Failure dataset for mortality prediction"

dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

if key in ws.datasets.keys():
    dataset = ws.datasets[key]
    print("The Dataset was found!")
else:
    # Create AML Dataset and register it into Workspace
    dataset = Dataset.Tabular.from_delimited_files(dataset_url)
    # Register Dataset in Workspace
    dataset = dataset.register(workspace=ws,
                               name=key,
                               description=description_text)

df = dataset.to_pandas_dataframe()
```

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning

In this section I have used a custom-coded model — a standard Scikit-learn Logistic Regression - which hyperparameters I optimized using HyperDrive.

Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

A Hyperdrive run is used to sweep over model parameters. The following steps are part of the process:

- Data preprocessing
- Splitting data into train and test sets
- Setting logistic regression parameters:
    - --C - Inverse of regularization strength
    - --max_iter - Maximum number of iterations convergence
- Azure Cloud resources configuration
- Creating a HyperDrive configuration using the estimator, hyperparameter sampler, and policy
- Retrieve the best run and save the model from that run

**RandomParameterSampling** 

Defines random sampling over a hyperparameter search space. In this sampling algorithm, parameter values are chosen from a set of discrete values or a distribution over a continuous range. This has an advantage against GridSearch method that runs all combinations of parameters and requires large amount of time to run.

For the Inverse of regularization strenght parameter I have chosen uniform distribution with min=0.0001 and max=1.0 For the Maximum number of iterations convergence I inputed a range of values (5, 25, 50, 100, 200, 500, 1000)

**BanditPolicy** 

Class Defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. This greatly helps to ensure if model with given parameters is not performing well, it is turned down instead of running it for any longer.

### Results
The best model given by HyperDrive resulted in training accuracy of **0.92%**. The hyperparameters of the model are as follows:

- --C = **0.544**
- --max_iter = **25**

**Improvement**
- One way to improve the result could be to change the range of hyperparameters to extend the search space.
- Other ways include changing the ML model completely or use a data set with much more data records if that would be a posibility



### Screenshots

**RunDetails widget**
<img src="images/???.png" width=75%>

**Best model**
<img src="images/???.png" width=75%>


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. This is to demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

