# Capstone Project: Azure Machine Learning Engineer

This project consists of two part: first part will be using Automated ML (denoted as AutoML) and the second part will use customized model whose hyperparameters are tuned using HyperDrive. The model trained by AutoML will be later on deployed and could be used as a ML service with which we can interact using REST API.

We will use an external dataset to train the model to classify heart failure based on clinical records and we will levarege all the different tools available in the AzureML Studio to do so. Please see the next section for more details about the dataset used in this project.




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

In this part of the project we make a use of Microsoft Azure Cloud to configure a cloud-based machine learning model and consequently deploy it. We first create a compute target with the following setting: `vm_size="Standard_D2_V2"`, `min_nodes=0`, `max_nodes=4` and then train a set of machine learning models leveraging AutoML to automaticaly train and tune a them using given target metric. In this case the selected target metric is `AUC_weighted`. A datastore retrieved by `data_store = ws.get_default_datastore()` is used to upload the dataset used to train the ML model and it is registered by using the following command  

```python
# Create AML Dataset and register it into Workspace
dataset = Dataset.Tabular.from_delimited_files(dataset_url)
# Register Dataset in Workspace
dataset = dataset.register(workspace=ws,
                           name=key,
                           description=description_text)
```

### AutoML Configuration and Settings
As mentioned above in the dataset section we are dealing with a binary classification. Therefore the argument `task` is set to `classification` and the since we are predicting `DEATH_EVENT` we also need to set `label_column_name="DEATH_EVENT"`. The dataset itself specified in `training_data=train_ds` and the compute target that we provisioned is set with `compute_target=aml_compute`.

To help manage child runs and when they can be performed, we recommend you create a dedicated cluster per experiment, and match the number of `max_concurrent_iterations` of your experiment to the number of nodes in the cluster. This way, you use all the nodes of the cluster at the same time with the number of concurrent child runs/iterations you want.

Besides other arguments that are self-explanatory, to automate Feature engineering AzureML enables this through `featurization` that needs to be set to `True`. This way features that best characterize the patterns in the data are selected to create predictive models.

Here is the code to set and configure the AutoML experiment

```python
# AutoMl settings
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric": "AUC_weighted",
    "enable_early_stopping": True,
    "verbosity": logging.INFO
}

# AutoMl config
automl_config = AutoMLConfig(compute_target=aml_compute,
                             task="classification",
                             training_data=train_ds,
                             label_column_name="DEATH_EVENT",
                             n_cross_validations=5,
                             featurization="auto",
                             path=project_folder,
                             debug_log = "automl_errors.log",
                             **automl_settings                             
                            )
```

Once the AutoML experiment is completed, we then select the best model in terms of `AUC_weighted` out of all models trained and deploy it using Azure Container Instance (ACI). So that the model can then be consumed via a REST API.



### Results

The best performing model trained by AutoML was `VotingEnsemble` with `AUC_weighted = 0.935` as can be seen in the screenshot below.

Few parameters of its inner estimators are shown as well as its individual weights used to aggregate the estimators and integers designating the different estimators used in the Ensemble. For more comprehensive details please see [this](automl.ipynb) notebook.

<img src="images/AMLBest_Model.PNG" width=75%>
<img src="images/AMLBest_Model2.PNG" width=75%>

**Best model in Azure Portal**

<img src="images/AMLBest_Model_in_Azure.PNG" width=75%>

**RunDetails widget**

<img src="images/AMLRunDetails_01.PNG" width=75%>
<img src="images/AMLRunDetails_02.PNG" width=75%>

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

The best model parameters are retrieved by using this code:

```python
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['arguments']

print("Best Experiment Run:")
print(f" Best Run Id: {best_run.id}")
print(f" Accuracy: {best_run_metrics['Accuracy']}")
print(f" Regularization Strength: {best_run_metrics['Regularization Strength:']}")
print(f" Max iterations: {best_run_metrics['Max iterations:']}")
```

Giving the following output:

```python 
Best Experiment Run:
 Best Run Id: HD_7fd07718-c5c5-42f5-ae1f-bd2f62a56380_29
 Accuracy: 0.9166666666666666
 Regularization Strength: 0.5443967516166649
 Max iterations: 25
```

**Improvement**
- One way to improve the result could be to change the range of hyperparameters to extend the search space.
- Other ways include changing the ML model completely or use a data set with much more data records if that would be a posibility



### Screenshots

**RunDetails widget**

<img src="images/HPRunDetails1.PNG" width=75%>
<img src="images/HPRunDetails2.PNG" width=75%>
<img src="images/HPRunDetailsVis.PNG" width=75%>
<img src="images/HPExp.PNG" width=75%>


**Best model**

<img src="images/HPBestModel.PNG" width=75%>
<img src="images/HPBestModelPortal.PNG" width=75%>


## Model Deployment

Here we use the best ML model from AutoML experiment and deploy it using Azure Container Instance (ACI). The model can be consumed via a REST API.

We also included file containing the environment details `myenv.yml` to ensure reproducibility.

To query the endpoint we use a 3 samples from the dataset so that we can evaluate whether the model performed well on the input data and retured the right prediction

Here is a code snipet with a sample input we used to make a request to the model: 

```python
# 3 sets of data to score, so we get two results back
data_df = test_df.sample(n=3)
labels = data_df.pop('DEATH_EVENT')

# Convert to JSON string
input_data = json.dumps({"data": data_df.to_dict(orient='records')})
```

Here is how the sample input data `input_data` looks like showing its format that we used to make the request:

```python
{"data": [{"age": 42.0, "anaemia": 1, "creatinine_phosphokinase": 86, "diabetes": 0, "ejection_fraction": 35, "high_blood_pressure": 0, "platelets": 365000.0, "serum_creatinine": 1.1, "serum_sodium": 139, "sex": 1, "smoking": 1, "time": 201}, {"age": 49.0, "anaemia": 0, "creatinine_phosphokinase": 972, "diabetes": 1, "ejection_fraction": 35, "high_blood_pressure": 1, "platelets": 268000.0, "serum_creatinine": 0.8, "serum_sodium": 130, "sex": 0, "smoking": 0, "time": 187}, {"age": 65.0, "anaemia": 0, "creatinine_phosphokinase": 56, "diabetes": 0, "ejection_fraction": 25, "high_blood_pressure": 0, "platelets": 237000.0, "serum_creatinine": 5.0, "serum_sodium": 130, "sex": 0, "smoking": 0, "time": 207}]}
```

And this snippet shows how we make a post request with the input data:

```python
# URL for the web service
scoring_uri = service.scoring_uri

# Set the content type
headers = {"Content-Type": "application/json"}

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
```

The output compared to the groundtrough labels from the dataset is shown here:

```python
print(f"Predictions from Service: {resp.json()}\n")
print(f"Data Labels: {labels.tolist()}")
```

```
Predictions from Service: [0, 0, 0]

Data Labels: [0, 0, 0]
```


**Active Endpoint**

<img src="images/AMLActive_Endpoint.PNG" width=75%>


## Screen Recording

This [screencast](https://youtu.be/TeFsuQAIhbI) shows the entire process of the working ML application and it demonstrates:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Future improvements

- One way to improve the ML model prediction capabilities would be to check whether the dataset is balanced, meaning if the data for both cases that we are trying to predict are equally represented in the dataset. Balancing data is a way to ensure that the ML model is learning both cases equally. In extreme case where one class is overrepresented in the dataset the ML model would almost always predict that specific class, as the model was not exposed enough to the other class. 

- We could also increase the `experiment_timeout_minutes` and let AutoML train for longer. 

- The primary metric used in the AutoML experiment is `AUC_weighted` and we could try if other metrices could be a better choice to train the model.

- Another way to improve the ML model could be to try train the model using Deep learning techniques and see if that is a good aproach to take. 
 
