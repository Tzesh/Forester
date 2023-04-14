# Forester
Modular python class to use Random Forest Classifier and make predictions without re-training data. Does search to find best suitable hyper parameters to the given dataset. Evaluates and saves the statistics, also logs every single action using a logging mechanism.

## Features
- [x] Doesn't need to re-train data
- [x] Don't need to manually preprocess data
- [x] Predicts using the best hyper parameters
- [x] Saves statistics
- [x] Logs every single action
- [x] Modular
- [x] Easy to use

## General Project Structure
- [x] data
    - [x] data.csv # your data that you want to use to train the model
- [x] log
    - [x] data_unique_datetime_identifier.txt # simply a log file
- [x] model
    - [x] data_model_encoders.pickle # encoders that are used to encode the data in the preprocessing step
    - [x] data_model_feature_names.pickle # feature names
    - [x] data_model_value_name.pickle # output name
- [x] statistics
    - [x] data_model_datetime_identifier_confusion_matrix.png # confusion matrix
    - [x] data_model_datetime_identifier_decision_tree.dot # decision tree of the first tree
    - [x] data_model_datetime_identifier_statistics.txt # statistics like accuracy, precision, recall, f1-score, etc.

## Usage
```python
from forester import Forester

# Initialize Forester
## Assumes that the data is in the './data/data.csv' file and the default delimiter is ','
## When we set train=True, it will train the model and save the required files
forester = Forester(train=True)

# Create your prediction data
val = [0,...,'Example', 1]

# call make_prediction method
## It will return the prediction
prediction = forester.make_prediction(val)

# Print the prediction
print(prediction)
```

Example usages from different datasets can be found in the Example.py file.

### When you first run the code (Example.py), it will train the model and save the required files. After that, it will use the saved files to make predictions without re-training the model.

![First run of Example.py](https://imgur.com/Ys7GDpE.png)

### On the sequential runs, it will use the saved files to make predictions without re-training the model.
![Second run of Example.py](https://imgur.com/S1Q45Do.png)

## Requirements
- [x] Python 3.6+
- [x] Scikit-learn
- [x] Pandas
- [x] NumPy
- [x] SciPy