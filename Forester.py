## import libraries

# data processing
import pandas as pd
import numpy as np

# data saving
import os
import pickle

# modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint

# visualization
from sklearn.tree import export_graphviz

# logging
import logging
import time
import sys

# define class Forester
class Forester():
    """
    Forester is a class that can be used to train and evaluate a model to predict failure mode of a given part number

    Parameters
    ----------
    path : str
        Path to csv file
    data_file_name : str
        Name of csv file
    header : int
        Header of csv file
    delimiter : str
        Delimiter of csv file
    model_directory : str
        Path to save model
    model_file_name : str
        Name of model file
    train : bool
        If True, model will be trained
    train_size : float
        Train size
    random_state : int
        Random state
    force_train : bool
        If True, model will be trained even if model is found
    evaluate_model : bool
        If True, model will be evaluated

    Attributes
    ----------
    path : str
        Path to csv file
    header : int
        Header of csv file
    delimiter : str
        Delimiter of csv file
    file_name : str
        Path to save model
    train : bool
        If True, model will be trained
    model_found : bool
        If True, model is found
    train_size : float
        Train size
    random_state : int
        Random state
    force_train : bool
        If True, model will be trained even if model is found
    evaluate_model : bool
        If True, model will be evaluated
    logger : logging.Logger
        Logger
    feature_names : list
        List of feature names
    """
    
    def __init__(self, path = './data/data.csv', header = 0, delimiter = ',', model_directory = './model/', train = False, test_size = 0.3, random_state = 1, force_train = False, evaluate_model = True):
        self.path = path # path to csv file
        self.data_file_name = path.split('/')[-1].split('.')[0] # name of csv file
        self.header = header # header of csv file
        self.delimiter = delimiter # delimiter of csv file
        self.model_directory = model_directory # path to save model
        self.model_file_name = self.data_file_name + '_model' # name of model file
        self.train = train # if True, model will be trained
        self.model_found = False # if True, model is found
        self.test_size = test_size # train size
        self.random_state = random_state # random state
        self.force_train = force_train # if True, model will be trained even if model is found
        self.evaluate_model = evaluate_model # default True, model will be evaluated
        self.encoders = {} # dictionary of label encoders
        self.__configure_logger() # configure logger
        self.__get_data() # get data

    def __configure_logger(self):
        """
        Configures logger
        """

        # create logger
        logger = logging.getLogger('')
        
        # set log level
        logger.setLevel(logging.DEBUG)

        # create prefix to log file
        prefix = time.strftime('%Y%m%d_%H%M%S')

        # create file handler and set level to debug
        fh = logging.FileHandler('./log/' + self.data_file_name + '_' + prefix + '.log')

        # create stream handler and set level to debug
        sh = logging.StreamHandler(sys.stdout)

        # create formatter
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

        # add formatter to fh and sh
        fh.setFormatter(formatter)

        # add formatter to fh and sh
        sh.setFormatter(formatter)

        # add fh and sh to logger
        logger.addHandler(fh)

        # add fh and sh to logger
        logger.addHandler(sh)

        # set logger
        self.logger = logger
    
    def _log(self, message, type = 'info'):
        """
        Logs message to console and file

        Parameters
        ----------
        message : str
            Message to be logged
        type : str
            Type of message to be logged
        """

        # if type is info
        if (type == 'info'):
            # log info
            self.logger.info(message)

        # if type is error
        elif (type == 'error'):
            # log error
            self.logger.error(message)

        # if type is warning
        elif (type == 'warning'):
            # log warning
            self.logger.warning(message)
        
        # if type is debug
        elif (type == 'debug'):
            # log debug
            self.logger.debug(message)

    def __get_data(self):
        """
        Loads data, preprocesses data, and trains model
        """

        # if force_train is True
        if (self.force_train):
            # load data
            df = self.__load_data()
            # preprocess data
            self.__preprocess_data(df)
            # train model
            self.__train_model()

        # check if there is file at path 'model/model.pkl'
        file_exists = os.path.isfile(self.model_directory + self.model_file_name + '.pickle')

        # if file exists
        if (file_exists):
            # load model
            self.__load_model()
        
        # if file does not exist
        else:
            # print model not found
            self._log('Pre-trained model is not found', 'error')
            # and if train is True
            if (self.train):
                # load data
                df = self.__load_data()
                # preprocess data
                self.__preprocess_data(df)
                # train model
                self.__train_model()

    def __load_data(self):
        """
        Loads data from csv file
        """

        # get data from csv file
        df = pd.read_csv(self.path, header = self.header, delimiter=self.delimiter)

        # drop rows with missing values
        df = df.dropna()

        # get features
        feature_names = df.columns

        # remove latest element of feature_names
        feature_names = feature_names[:-1]

        # value name
        value_name = df.columns[-1]

        # save value name
        self.value_name = value_name

        # save feature names
        self.feature_names = feature_names

        # return df
        return df
    
    def __load_model(self):
        """
        Loads model, ptvl_encoder, prefix_encoder, base_encoder, suffix_encoder, and failure_mode_encoder
        """

        # load model
        self.model = pickle.load(open(self.model_directory + self.model_file_name + '.pickle', 'rb'))

        # load encoders
        self.encoders = pickle.load(open(self.model_directory + self.model_file_name + '_encoders.pickle', 'rb'))

        # load feature names
        self.feature_names = pickle.load(open(self.model_directory + self.model_file_name + '_feature_names.pickle', 'rb'))

        # load value name
        self.value_name = pickle.load(open(self.model_directory + self.model_file_name + '_value_name.pickle', 'rb'))

        # set model_found to True
        self.model_found = True

        # print out model found
        self._log('Model found and loaded')
    
    def __preprocess_data(self, df):
        """
        Preprocesses data, saves encoders, and saves data as a class variable

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing data
        """

        # loop foreach df column
        for column in df.columns:

            # get first element of column
            first_element = df[column].iloc[0]

            # check if first element is a string
            if (isinstance(first_element, str)):
                # if whole column is string then create a new encoder
                self.encoders[column] = LabelEncoder()

                # convert column to numerical data
                df[column] = self.encoders[column].fit_transform(df[column])

        # save preprocessed data
        self.df = df

        # print out data has been preprocessed
        self._log('Data has been preprocessed')
    
    def make_prediction(self, val):
        """
        Makes a prediction with a given single value

        Parameters
        ----------
        val : numpy array
            Array of categorical data to be predicted
        
        Returns
        -------
        predicted : numpy array
            Array of predicted categorical data
        """

        # check if val length is equal to feature_names length
        if (len(val) != len(self.feature_names)):
            # print out error
            self._log('Error: Length of val is not equal to length of feature_names', 'error')
            # return None
            return None

        # making prediction print out
        self._log(('Making prediction for: ', val))

        # convert val to numerical data
        val = self.__convert_to_numerical(val)

        # print out converted to numerical data
        self._log(('Converted to numerical data: ', val))

        # make a single prediction
        prediction = self.model.predict(np.array(val).reshape(1, -1))

        # convert prediction to categorical data
        predicted = self.__convert_numerical_to_categorical(prediction)

        # print out predicted value
        self._log(('Prediction: ', predicted))

        # return predicted value
        return predicted
    
    def __convert_to_numerical(self, val):
        """
        Converts categorical data to numerical data

        Parameters
        ----------
        val : numpy array
            Array of categorical data

        Returns
        -------
        val : numpy array
            Array of numerical data
        """

        # initialize feature index
        feature_index = 0

        # loop foreach val
        for i in range(len(val)):
            # get val[i]
            value = val[i]

            # check if value is a string
            if (isinstance(value, str)):

                # get feature name
                feature_name = self.feature_names[feature_index]

                # convert value to numerical data
                val[i] = self.encoders[feature_name].transform([value])[0]

            # increment feature index
            feature_index += 1

            # continue

        # return val
        return val
    
    def __convert_numerical_to_categorical(self, val):
        """
        Converts numerical data to categorical data

        Parameters
        ----------
        val : numpy.ndarray
            The numerical data

        Returns
        -------
        val : numpy.ndarray
            The categorical data
        """

        print(self.encoders)

        # check if encoders is None
        if (self.encoders.__len__() == 0):
            # simply return val
            return val

        # get value encoder
        encoder = self.encoders[self.value_name]

        # if encoder is None
        if (encoder is None):
            # simply return val
            return val

        # convert val to categorical data
        val = encoder.inverse_transform(val)

        # return val
        return val

    def __split_data(self):
        """
        Splits data into train and test sets

        Returns
        -------
        X_train : pandas.core.frame.DataFrame
            The training data
        X_test : pandas.core.frame.DataFrame
            The testing data
        y_train : pandas.core.series.Series
            The training labels
        y_test : pandas.core.series.Series
            The testing labels
        """

        # split data into X and y
        X = self.df.drop([self.value_name], axis=1)
        y = self.df[self.value_name]

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # print out data has been split
        self._log(('Data has been split with train size: ', self.test_size, ' and random state: ', self.random_state))

        # return X_train, X_test, y_train, y_test
        return X_train, X_test, y_train, y_test
    
    def __randomize_search(self, X_train, y_train):
        """
        Does a randomized search for the best hyperparameters for the model

        Parameters
        ----------
        X_train : pandas.core.frame.DataFrame
            The training data
        y_train : pandas.core.series.Series
            The training labels
        
        Returns
        -------
        dict
            The best hyperparameters
        """

        # define the model
        model = RandomForestClassifier()

        # define the space of hyperparameters to search
        space = dict()

        # define hyperparameters
        space['n_estimators'] = randint(100, 1000)
        space['max_features'] = randint(1, self.feature_names.__len__() + 1)

        # define search
        search = RandomizedSearchCV(model, space, n_iter=100, scoring='accuracy', n_jobs=-1, random_state=1)

        # execute search
        result = search.fit(X_train, y_train)

        # summarize result
        self._log('Randomized Search has been completed')

        # print best score
        self._log(('Best Score: %s' % result.best_score_))

        # print best hyperparameters
        self._log(('Best Hyperparameters: %s' % result.best_params_))

        # return best hyperparameters
        return result.best_params_
    
    def __train_model(self):
        """
        Trains the model and calls the __save_model() method to save the model
        Also if evaluate_model is True, it calls the __evaluate_model() method to evaluate the model
        """

        # get required parameters
        X_train, X_test, y_train, y_test = self.__split_data()
        
        # get best hyperparameters
        best_params = self.__randomize_search(X_train, y_train)

        # define the model
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_features=best_params['max_features'])

        # fit the model on the whole dataset
        model.fit(X_train.values, y_train)

        # save model
        self.model = model

        # save model
        self.__save_model()

        # return if evaluate_model is False
        if (self.evaluate_model == False):
            # print model is not evaluated due to evaluate_model being False
            self._log('Model is not evaluated due to evaluate_model being False')
            return

        # evaluate model
        self.__evaluate_model(X_test, y_test)
    
    def __save_model(self):
        """
        Save model and encoders to pickle files in a folder called 'models'
        in the current working directory.

        """

        # determine operation
        operation = 'wb' if self.model_found == False else 'x'

        # save model
        with open(self.model_directory + self.model_file_name + '.pickle', operation) as f:
            pickle.dump(self.model, f)
        
        # save ptvl_encoder
        with open(self.model_directory + self.model_file_name + '_encoders.pickle', operation) as f:
            pickle.dump(self.encoders, f)

        # save feature names
        with open(self.model_directory + self.model_file_name + '_feature_names.pickle', operation) as f:
            pickle.dump(self.feature_names, f)
        
        # save value name
        with open(self.model_directory + self.model_file_name + '_value_name.pickle', operation) as f:
            pickle.dump(self.value_name, f)

        # print out model is trained
        self._log('Model has been trained and saved')

    def __evaluate_model(self, X_test, y_test):
        """
        Evaluates the model and prints out the accuracy and confusion matrix
        Saves the confusion matrix as a png file
        Lastly calls plot decision tree function to plot the decision tree

        Parameters
        ----------
        X_test : pandas.core.frame.DataFrame
            Test data
        y_test : pandas.core.series.Series
            Test labels
        """

        # make a single prediction
        yhat = self.model.predict(X_test)

        # print model is evaluated
        self._log('Model has been evaluated')

        # evaluate predictions
        accuracy = accuracy_score(y_test, yhat)
        self._log(('Accuracy: %.2f' % (accuracy*100)))

        # confusion matrix
        cm = confusion_matrix(y_test, yhat)

        # plot confusion matrix
        cm_display = ConfusionMatrixDisplay(cm).plot()

        # create a timestamp
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # save confusion matrix
        cm_display.figure_.savefig('./statistics/' + self.data_file_name + '_' + timestamp + '_confusion_matrix.png')

        # print confusion matrix
        self._log('Confusion Matrix: ')
        self._log(cm)

        # precision
        precision = precision_score(y_test, yhat, average='weighted')
        self._log(('Precision: %.2f' % (precision*100)))

        # recall
        recall = recall_score(y_test, yhat, average='weighted')
        self._log(('Recall: %.2f' % (recall*100)))

        # feature importance
        importance = self.model.feature_importances_
        self._log(('Feature Importance: %s' % importance))

        # save statistics as a text file
        with open('./statistics/' + self.data_file_name + '_' + timestamp + '_statistics.txt', 'w') as f:
            f.write('Accuracy: %.2f' % (accuracy*100) + ' %' + ' Precision: %.2f' % (precision*100) + ' %' + ' Recall: %.2f' % (recall*100) + ' %')

        # call __plot_decision_tree
        self.__plot_decision_tree(timestamp)
    
    def __plot_decision_tree(self, timestamp):
            """
            This function plots the first decision tree of the random forest classifier
            and saves it as a .dot file.

            Parameters
            ----------
            timestamp : str
                Timestamp of the current time when evaluation function is called
            """
            
            # name output file
            out_file = './statistics/' + self.data_file_name + '_' + timestamp + '_decision_tree.dot'

            # export as dot file
            export_graphviz(self.model.estimators_[0], out_file=out_file, feature_names = self.feature_names, class_names = self.value_name, rounded = True, proportion = False, precision = 2, filled = True)

            # you can convert the .dot file to a .png or .pdf using this command:
            # but you need to install the graphviz package first
            # dot -Tpng decision_tree.dot -o decision_tree.png
            # dot -Tpdf decision_tree.dot -o decision_tree.pdf

            # print decision tree
            self._log('First Decision Tree has been plotted and saved')
    
    # destructor
    def __del__(self):
        """
        Destructor
        """

        # create prefix to log file
        prefix = time.strftime('%Y%m%d_%H%M%S')

        # create file handler and set level to debug
        fh = logging.FileHandler('./log/' + self.data_file_name + '_' + prefix + '.log')

        # create stream handler and set level to debug
        sh = logging.StreamHandler(sys.stdout)

        # close stream handler
        sh.close()

        # close file handler
        fh.close()

        # remove all handlers
        self.logger.handlers = []

        # print object is destroyed
        self._log('Forester (' + self.data_file_name +  ') is destroyed')