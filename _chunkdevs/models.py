import os
import joblib as jl

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import tqdm

import torch
import torch.optim as optim
import tensorflow as tf

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator

from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

from tensorflow import keras
from tensorflow.keras import layers, utils # type: ignore
from tensorflow.keras.initializers import GlorotUniform # type: ignore

from tensorflow.keras.layers import Dropout, TextVectorization, Embedding, Bidirectional, LSTM # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

from typing import List, Dict, Any, Tuple, Union, Optional


class ModelHandler:
    """
    A class for handling the saving and loading of models.

    Attributes
    ----------
    `directory` : str
        The directory where models will be saved and loaded from.

    Methods
    -------
    * `save_model(model, name)`
        Saves a single model to the specified directory.
    * `load_model(name)`
        Loads a single model from the specified directory.
    * `save_models(models)`
        Saves multiple models to the specified directory.
    * `load_models(names)`
        Loads multiple models from the specified directory.
    """
    def __init__(self, directory: str) -> None:
        """
        Initializes the ModelHandler with the specified directory.
        """
        self.directory = directory

        # Create the directory if it does not exist.
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_model(self, model: BaseEstimator, name:str) -> None:
        """
        Saves a single model to the specified directory.

        :param `model`: The trained model to be saved. 
        :param `name`: The name of the model file without extension.
        """
        # Create the full path for the model file.
        model_path = os.path.join(self.directory, f"{name}.pkl")

        # Save the model to the specified path.
        jl.dump(model, model_path)

        # Print a confirmation message with the model name and path.
        response = f"| Save model '{name}' to {model_path} |"
        print("-" * len(response))
        print(response)
        print("-" * len(response))

    def load_model(self, name: str) -> BaseEstimator:
        """
        Loads a single model from the specified directory.

        :param `name`: The name of the model file without extension.
        :return: The loaded specified model.
        :raises `FileNotFoundError`: If the model file does not exist in the specified directory.
        """
        # Create the full path for the model file.
        model_path = os.path.join(self.directory, f"{name}.pkl")

        # Raise an error if the model file does not exist.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model '{name}' found at {model_path}")
        
        # Load the model from the specified path.
        model = jl.load(model_path)

        return model
    
    def save_models(self, models: Dict[str, BaseEstimator]) -> None:
        """
        Save multiple models to the specified directory.

        :param `models`: A dictionary where keys are model names and values are the models to be saved.
        """
        for name, model in models.items():
            # Save each model in the dictionary.
            self.save_model(model, name)
    
    def load_models(self, names: List[str]) -> Dict[str, BaseEstimator]:
        """
        Load multiple models from the specified directory.

        :param `names`: A list of model names to be loaded without extension.
        :return: A dictionary where keys are model names and values are the loaded models.
        """
        models = {}
        for name in names:
            # Load each model specified in the list.
            models[name] = self.load_model(name)

        # Return the dictionary of loaded models.
        return models
    

class ChunkDevs_GridTrainer:
    """
    A class for training models using Grid Search with cross-validation.

    Attributes
    ----------
    model : BaseEstimator
        The model to be optimized.
    param_grid : Dict[str, List[Any]]
        The parameter grid to search over.
    scoring : str
        The scoring metric to use for evaluation.
    cv : Union[int, KFold]
        The cross-validation strategy.
    n_jobs : int
        The number of jobs to run in parallel.
    grid_search : GridSearchCV
        The GridSearchCV object after fitting.

    Methods
    -------
    train(X_train, y_train)
        Train the model using Grid Search with the provided training data.
    predict(X_test)
        Make predictions using the trained model.
    get_best_params()
        Get the best parameters found by Grid Search.
    get_best_score()
        Get the best score achieved by Grid Search.
    get_classification_report(X_test, y_test)
        Get the classification report for the test data.
    get_confusion_matrix(X_test, y_test)
        Get the confusion matrix for the test data.
    save_results(X_test, y_test, path)
        Save the Grid Search results and classification report to a file.
    """
    
    def __init__(self, model: BaseEstimator, param_grid: Dict[str, List[Any]], 
                 scoring: str, cv: Union[int, KFold], n_jobs: int) -> None:
        """
        Initialize the ChunkDevs_GridTrainer with the specified parameters.
        """
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.grid_search = None
        
    def __repr__(self) -> str:
        """
        Return a string representation of the ChunkDevs_GridTrainer.
        """
        if self.grid_search:
            return (f"ChunkDevs_GridTrainer(model={self.model}, "
                    f"params={self.param_grid}, scoring={self.scoring}, cv={self.cv})")
        else:
            return "ChunkDevs_GridTrainer(model='untrained')"
    
    def __getattr__(self, attr):
        """
        Delegate attribute access to the GridSearchCV object if it exists.

        :param attr: The attribute to access.
        :raises AttributeError: If the attribute does not exist.
        """
        if self.grid_search and hasattr(self.grid_search, attr):
            return getattr(self.grid_search, attr)
        raise AttributeError(f"'ChunkDevs_GridTrainer' object has no attribute '{attr}'")
        
    def train(self, feature_data: np.ndarray, target_data: np.ndarray) -> BaseEstimator:
        """
        Train the model using Grid Search with the provided training data.

        :param feature_data: The training feature data.
        :param target_data: The training target data.
        :return: The best estimator found by Grid Search.
        """
        # Check if cv is an integer or a KFold instance
        if not isinstance(self.cv, (int, KFold)):
            # Raise an error if cv is not an integer or a KFold instance
            raise ValueError("cv must be an integer or a KFold instance.")
        
        # Perform grid search using GridSearchCV with specified parameters
        self.grid_search = GridSearchCV(
            estimator=self.model, 
            param_grid=self.param_grid, 
            scoring=self.scoring, 
            cv=self.cv, 
            n_jobs=self.n_jobs
        )

        # Fit the grid search to the training data
        self.grid_search.fit(feature_data, target_data)

        # Return the best estimator found by grid search
        return self.grid_search.best_estimator_
    
    def predict(self, feature_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        :param feature_data: The test feature data.
        :return: The predicted labels.
        :raises ValueError: If the model has not been trained.
        """
        # Check if grid search has been performed
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("You must train the model before making predictions.")

        # Make predictions using the best estimator from grid search on the test data
        return self.grid_search.predict(feature_data)
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found by Grid Search.

        :return: The best parameters as a dictionary.
        :raises ValueError: If the model has not been trained.
        """
        # Check if grid search has been performed
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("Grid search has not been run. Train the model first.")
        
        # Return the best parameters found by grid search
        return self.grid_search.best_params_
    
    def get_best_score(self) -> float:
        """
        Get the best score achieved by Grid Search.

        :return: The best score as a float.
        :raises ValueError: If the model has not been trained.
        """
        # Check if grid search has been performed
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("Grid search has not been run. Train the model first.")
        
        # Return the best score obtained from grid search
        return self.grid_search.best_score_
    
    def get_classification_report(self, feature_data: np.ndarray, target_data: np.ndarray) -> str:
        """
        Get the classification report for the test data.

        :param feature_data: The test feature data.
        :param target_data: The test target data.
        :return: The classification report as a string.
        :raises ValueError: If the model has not been trained.
        """
        # Check if grid search has been performed
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("You must train the model before printing the classification report.")
        
        # Predict the labels using the trained model on the test data
        y_pred = self.predict(feature_data)

        # Generate and return the classification report based on true and predicted labels
        return classification_report(target_data, y_pred)
    
    def get_confusion_matrix(self, feature_data: np.ndarray, target_data: np.ndarray) -> np.ndarray:
        """
        Get the confusion matrix for the test data.

        :param feature_data: The feature data.
        :param target_data: The target data.
        :return: The confusion matrix as a numpy array.
        :raises ValueError: If the model has not been trained.
        """
        # Check if grid search has been performed
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("You must train the model before printing the confusion matrix.")
        
        # Predict the labels using the trained model on the test data
        y_pred = self.predict(feature_data)

        # Generate and return the confusion matrix based on true and predicted labels
        return confusion_matrix(target_data, y_pred)
    
    def save_results(self, X_test: np.ndarray, y_test: np.ndarray, path: str) -> None:
        """
        Save the Grid Search results and classification report to a file.

        :param X_test: The test feature data.
        :param y_test: The test target data.
        :param path: The path to save the results.
        :raises ValueError: If the model has not been trained.
        """
        # Check if the grid search has been performed.
        if not self.grid_search:
            # Raise an error if grid search has not been performed
            raise ValueError("You must train the model before saving the report.")
        
        # Create directories as needed to save the report file
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get the best parameters found by grid search
        best_params = self.get_best_params()
        # Get the best score obtained from grid search
        best_score = self.get_best_score()
        # Generate the classification report using test data
        report = self.get_classification_report(X_test, y_test)

        # Write the report contents to a file
        with open(path, 'a') as file:
            file.write(f"Model: {self.model}\n")
            file.write(f"Best Parameters: {best_params}\n")
            file.write(f"Best Score: {best_score}\n")
            file.write(f"Classification Report:\n{report}\n")

        # Print a confirmation message.
        print(f"Report saved to {path}")



class ChunkDevs_LSTM:
    """
    A class for building and training an RNN model for text classification.

    Attributes
    ----------
    `vocab_size` : int
        The size of the vocabulary.
    `sequence_length` : int
        The length of the input sequences.
    `embedding_dim` : int
        The dimension of the embedding layer.
    `lstm_units` : Optional[int]
        The number of LSTM units.
    `dropout_rate` : float
        The dropout rate for regularization.
    `val_size` : float
        The proportion of validation data.
    `random_state` : int
        The random seed for reproducibility.
    `metrics` : str
        The type of metrics ('binary' or 'multiclass').
    `model` : Optional[Model]
        The Keras model for the LSTM.
    `vectorizer` : TextVectorization
        The text vectorization layer for preprocessing.

    Methods
    -------
    * `prepare_data(dev_texts, test_texts, dev_labels, test_labels)`
        Prepare the training, validation, and test data.

    * `prepare_model()`
        Prepare the LSTM model architecture.

    * `build_model()`
        Build and compile the LSTM model.

    * `train(X_train y_train, X_val, y_val, batch_size, epochs)`
        Train the LSTM model.

    * `evaluate(X_test, y_test)`
        Evaluate the LSTM model on test data.

    * `predict(texts)`
        Make predictions using the LSTM model.

    * `get_classification_report(X_test, y_test)`
        Get the classification report for the test data.

    * `get_confusion_matrix(X_test, y_test)`
        Get the confusion matrix for the test data.

    * `save_results(X_test, y_test, path)`
        Save the classification report to a file
    """
    def __init__(self,
                vocab_size: int, sequence_length: int, embedding_dim: int,
                lstm_units: Optional[int], dropout_rate: float, val_size: float,
                random_state: int, metrics: str) -> None:
        """
        Initialize the ChunkDevs_RNN with the specified parameters.
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics
        self.model = None
        self.vectorizer = TextVectorization(max_tokens=self.vocab_size, standardize="lower", 
                                            output_mode="int", output_sequence_length=self.sequence_length)


    def __repr__(self) -> str:
        """
        Return a string representation of the ChunkDevs_LSTM.
        """
        if self.model:
            return f"ChunkDevs_LSTM(\n model='{self.model}' \n vectorizer='{self.vectorizer}')"
        else:
            return "ChunkDevs_LSTM(model='untrained')"

    def prepare_data(self, X_dev: pd.Series, X_test: pd.Series, 
                    y_dev: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the training, validation, and test data.

        :param `dev_texts`: The development texts.
        :param `test_texts`: The test texts.
        :param `dev_labels`: The development labels.
        :param `test_labels`: The test labels.
        :return: The prepared data as numpy arrays.
        """

        # Split the development data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=self.val_size, random_state=self.random_state)



        # Adapt the vectorizer to the development texts
        self.vectorizer.adapt(X_dev)
        self.vectorizer.adapt(X_train)
        self.vectorizer.adapt(X_val)
        self.vectorizer.adapt(X_test)

        # Vectorize the development and test texts using the adapted vectorizer
        text_vect_dev = self.vectorizer(X_dev)
        text_vect_train = self.vectorizer(X_train)
        text_vect_val = self.vectorizer(X_val)
        text_vect_test = self.vectorizer(X_test)

        # Convert the vectorized development and test texts to a NumPy array
        X_dev = text_vect_dev.numpy()
        X_train = text_vect_train.numpy()
        X_val = text_vect_val.numpy()
        X_test = text_vect_test.numpy()

        # Convert the development and test labels to categorical format
        y_dev = to_categorical(y_dev)
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)

        

        return X_dev, y_dev, X_train, y_train, X_val, y_val, X_test, y_test
    
    def prepare_model(self) -> Tuple[layers.Input, layers.Layer]:
        """
        Prepare the LSTM model architecture.

        :return: The input and output layers of the model.
        """
        # Create an input layer for the model with the specified sequence length and data type
        inputs = keras.Input(shape=(self.sequence_length,), dtype=tf.int64, name="inputs")
        # Create an embedding layer to convert input sequences into dense vectors
        embedded_sequences = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)

        # Check if the metrics type is binary
        if self.metrics == "binary":
            # Create a bidirectional LSTM layer with specified units and dropout rate
            lstm_output = layers.Bidirectional(layers.LSTM(self.lstm_units, dropout=self.dropout_rate))(embedded_sequences)
            # Create a dense layer with 8 units and ReLU activation
            dense_output = layers.Dense(8, activation="relu")(lstm_output)
        # Check if the metrics type is multiclass
        elif self.metrics == "multiclass":
            # Create a bidirectional LSTM layer with 64 units and Glorot uniform initializer
            lstm_output = layers.Bidirectional(layers.LSTM(64, kernel_initializer=GlorotUniform()))(embedded_sequences)
            # Apply dropout to the LSTM output
            lstm_output = Dropout(self.dropout_rate)(lstm_output)
            # Create a dense layer with 32 units and ReLU activation
            dense_output = layers.Dense(32, activation="relu")(lstm_output)
            # Apply dropout to the dense layer output
            dense_output = Dropout(0.2)(dense_output)
        else:
            # Raise an error if the metrics type is not supported
            raise ValueError(f"{self.metrics} not supported. Use 'binary' or 'multiclass'.")
        
        return inputs, dense_output
    
    def build_model(self) -> None:
        """
        Build and compile the LSTM model.
        """
        # Prepare the model inputs and the dense output layer
        inputs, dense_output = self.prepare_model()
        
        # Check if the metrics type is binary
        if self.metrics == "binary":
            # Create an output layer with 2 units and softmax activation for binary classification
            outputs = layers.Dense(2, activation="softmax")(dense_output)
            # Build the model using the inputs and outputs
            self.model = keras.Model(inputs, outputs)
            # Compile the model with binary cross-entropy loss and F1 score as the metric
            self.model.compile(optimizer="adam",
                               loss="binary_crossentropy",
                               metrics=["F1Score"])
        # Check if the metrics type is multiclass
        elif self.metrics == "multiclass":
            # Create an output layer with 6 units and softmax activation for multiclass classification
            outputs = layers.Dense(6, activation="softmax")(dense_output)
            # Build the model using the inputs and outputs
            self.model = keras.Model(inputs, outputs)
            # Compile the model with categorical cross-entropy loss and accuracy as the metric
            self.model.compile(optimizer="adam",
                               loss="categorical_crossentropy",
                               metrics=["accuracy"])
            

    def train(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None, 
          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
          batch_size: int = 32, epochs: int = 4) -> None:
        """
        Train or retrain the LSTM model.

        :param X_train: The training feature data.
        :param y_train: The training target data.
        :param X_val: The validation feature data.
        :param y_val: The validation target data.
        :param batch_size: The batch size for training.
        :param epochs: The number of training epochs.
        """
        # Check if the model has been built
        if self.model is None:
            # Build the model if it hasn't been built yet
            self.build_model()

            # Ensure training and validation data are provided for standard training
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("X_train, y_train, X_val, and y_val must be provided.")
            
        # Train the model on the training data with validation data
        self.history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

        # Print a message indicating that training has been completed
        print("Training completed.")

    def final_retrain(self, X_train: np.ndarray = None, y_train: np.ndarray = None, 
          batch_size: int = 32, epochs: int = 4) -> None:
        
        """
        Final retraining of the LSTM model.

        :param X_train: The training feature data.	
        :param y_train: The training target data.
        :param batch_size: The batch size for training.
        :param epochs: The number of training epochs.
        """

        if self.model is None:
        # Build the model if it hasn't been built yet
            self.build_model()
        
        self.history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        
        
    def evaluate(self, feature_data: np.ndarray, target_data: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the LSTM model on test data.

        :param `feature_data`: The feature data.
        :param `target_data`: The target data.
        :return: The evaluation loss and accuracy.
        :raises `ValueError`: If the model has not been trained.
        """
        # Check if the model has been trained
        if self.model is None:
            # Raise an error if the model has not been trained yet
            raise ValueError("The model has not been trained yet.")
        
        # Evaluate the model on the provided feature and target data
        return self.model.evaluate(feature_data, target_data)

    def predict(self, texts: np.ndarray) -> np.ndarray:
        """
        Make predictions using the LSTM model.

        :param `texts`: The input texts.
        :return: The predicted labels.
        :raises `ValueError`: If the model has not been trained.
        """
        # Check if model has been trained.
        if self.model is None:
            # Raise an error if the model has not been trained yet
            raise ValueError("The model has not been trained yet.")
        
        # Transform the input texts into vectors using the vectorizer and convert to a NumPy array

        # Use the trained model to make predictions on the vectorized texts
        return self.model.predict(texts)
    
    
    def get_classification_report(self, feature_data: np.ndarray, target_data: np.ndarray) -> str:
        """
        Get the classification report for the test data.

        :param `feature_data`: The feature data.
        :param `target_data`: The target data.
        :return: The classification report as a string.
        :raises `ValueError`: If the model has not been trained.
        """
        # Check if model has been trained.
        if self.model is None:
            # Raise an error if grid search has not been performed
            raise ValueError("The model has not been trained yet.")
        
        # Predict the labels using the trained model on the test data
        y_pred = np.argmax(self.model.predict(feature_data), axis=1)
        y_true = np.argmax(target_data, axis=1)
 
        return classification_report(y_true, y_pred)
    
    def get_confusion_matrix(self, feature_data: np.ndarray, target_data: np.ndarray) -> np.ndarray:
        """
        Get the confusion matrix for the test data.

        :param `feature_data`: The feature data.
        :param `target_data`: The target data.
        :return: The confusion matrix as a numpy array.
        :raises `ValueError`: If the model has not been trained.
        """
        # Check if model has been trained.
        if self.model is None:
            # Raise an error if grid search has not been performed
            raise ValueError("The model has not been trained yet.")
        
        # Predict the labels using the trained model on the test data
        y_pred = np.argmax(self.model.predict(feature_data), axis=1)
        y_true = np.argmax(target_data, axis=1)
        return confusion_matrix(y_true, y_pred)
        
    def save_results(self, X_test: np.ndarray, y_test: np.ndarray, path:str) -> None:
        """
        Save the classification report to a file.

        :param `X_test`: The test feature data.
        :param `y_test`: The test target data.
        :param `path`: The path to the file where the report will be saved.
        :raises `ValueError`: If the model has not been trained.
        """
        # Check if the model has been trained.
        if self.model is None:
            raise ValueError("You must train the model before saving the report.")
        
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Generate the classification report.
        report = self.get_classification_report(X_test, y_test)

        # Save the results to the specified path.
        with open(path, 'a') as file:
            file.write(f"Models: {self.model} \n\n")
            file.write(f"Best Parameters: \n")
            file.write(f"Classification Report: \n")
            file.write(f"{str(report)} \n\n")

        response = f"| Report saved to {path} |"

        # Print a confirmation message.
        print("-" * len(response))
        print(response)
        print("-" * len(response))

    

class ChunkDevs_Transformer(nn.Module):
    """
    A class that use PyTorch module for building and training Transformer-based models (BERT or RoBERTa) for text classification.

    Attributes
    ----------
    `model_type` : str
        The type of Transformer model ("bert" or "roberta").
    `sequences_length` : int
        The maximum length of input sequences.
    `classes` : int
        The number of classes for classification.
    `random_state` : int
        The random seed for reproducibility.
    `device` : torch.device
        The device to run the model on (CPU or GPU).

    Methods
    -------
    * `set_seed(seed)`
        Set seed for reproducibility across runs.

    * `forward(input_ids, attention_mask)`
        Forward pass of the model.

    * `train_model(dataloader, optimizer, scheduler, num_epochs)`
        Train the Transformer model.

    * `evaluate_model(dataloader, labels_name)`
        Evaluate the Transformer model on validation or test data.

    * `predict(texts)`
        Make predictions using the trained Transformer model.

    * `save_results(dataloader, path, labels_name)`
        Save evaluation results (accuracy and classification report) to a file.

    * `train_val_curves(train_dataloader, val_dataloader, optimizer, scheduler, plot_epochs, title)`
        Draw the loss/ accurracy curves of the transformer
    """
    def __init__(self, model_type: str, sequences_length: int, classes: int, random_state: int) -> None:
        """
        Initialize the ChunkDevs_Transformer with the specified parameters.
        """
        super(ChunkDevs_Transformer, self).__init__()
        self.model_type = model_type
        self.sequences_length = sequences_length
        self.classes = classes
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_seed(self.random_state)

        if model_type == "bert":
            self.model_name = "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.model.config.hidden_size, classes)
        elif model_type == "roberta":
            self.model_name = "roberta-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaModel.from_pretrained(self.model_name)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.model.config.hidden_size, classes)
        elif model_type == "bert_multilingual":
            self.model_name = "bert-base-multilingual-cased"
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.model.config.hidden_size, classes)
        else:
            raise ValueError(f"Unsupported model: {model_type}")
        
        self.to(self.device)
    
    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set seed for reproducibility across runs.

        :param `seed`: The seed value to set.
        """
        # Set the PYTHONHASHSEED environment variable to control hash seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Set the random seed for Python's built-in random module
        random.seed(seed)

        # Set the random seed for NumPy
        np.random.seed(seed)

        # Set the random seed for PyTorch on the CPU
        torch.manual_seed(seed)

        # If CUDA is available, set the random seed for all GPUs
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __repr__(self) -> str:
        """
        Return a string representation of the ChunkDevs_Transformer.
        """
        if self.model:
            return f"ChunkDevs_Transformer(model_type='{self.model_type}', model_name='{self.model_name}', \nmodel='{self.model}', \ntokenizer='{self.tokenizer}', \nclassifier='{self.classifier}')"
        else:
            return f"ChunkDevs_Transformer(model='untrained')"

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param `input_ids`: The input IDs representing the sequences.
        :param `attention_mask`: The attention mask for masking out padded tokens.
        :return: The output logits from the classifier.
        """
        # Forward pass through the model to get outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Check if the model type is "bert"
        if self.model_type == "bert":
            # Extract the pooled output from the BERT model
            pooled_outputs = outputs.pooler_output
        else:
            # Extract the pooled output from the BERT model
            pooled_outputs = outputs[0][:, 0, :]
        
        # Apply dropout to the pooled outputs
        x = self.dropout(pooled_outputs)

        # Pass the dropout output through the classifier to get logits
        logits = self.classifier(x)

        return logits
    
    def train_model(self, dataloader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, num_epochs: int) -> None: 
        """
        Train the Transformer model.

        :param `dataloader`: The DataLoader providing the training data.
        :param `optimizer`: The optimizer for gradient descent.
        :param `scheduler`: The learning rate scheduler.
        :param `num_epochs`: The number of training epochs.
        """
        # Set the model to training mode
        self.train()

        # Define the loss criterion as Cross Entropy Loss
        criterion = nn.CrossEntropyLoss()
        train_losses = []

        # Training loop for the specified number of epochs
        for epoch in range(num_epochs):
            total_loss = 0
            # Create a progress bar for visualization
            with tqdm.tqdm(total=len(dataloader.dataset), desc=f"Training Epoch {epoch+1}/{num_epochs}: ", colour="blue", ncols=100) as pbar:
                # Iterate over batches in the dataloader
                for batch in dataloader:

                    # Zero the gradients of the optimizer
                    optimizer.zero_grad()

                    # Move batch tensors to the appropriate device (CPU or GPU)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    # Pass the input data through the model and calculate the loss
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                    # Perform backward pass to compute gradients, update the model weights and learning rate scheduler
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    # Update the total loss with the current batch loss
                    total_loss += loss.item()

                    # Update the progress bar to reflect batch processing
                    pbar.update(dataloader.batch_size)

            # Calculate the average loss for the current epoch
            average_loss = total_loss / len(dataloader)
            train_losses.append(average_loss)

            print(f"Train Loss for Epoch {epoch+1}: {average_loss:.4f}")
        print("Trained Completed.")
    
    def evaluate_model(self, dataloader: DataLoader, labels_name: Optional[List[str]] = None) -> Tuple[float, str]:
        """
        Evaluate the Transformer model on validation or test data.

        :param `dataloader`: The DataLoader providing the evaluation data.
        :param `labels_name`: The names of the target labels (optional).
        :return: The accuracy and classification report as a string.
        """
        # Set the model to evaluation mode
        self.eval()
        predictions = []
        actual = []

        # Perform evaluation without gradient calculations
        with torch.no_grad():
            # Create a progress bar for visualization
            with tqdm.tqdm(total=len(dataloader.dataset), desc="EVALUATION:", colour="green", ncols=100) as pbar:
                # Iterate over batches in the dataloader
                for batch in dataloader:
                    # Move batch tensors to the appropriate device (CPU or GPU)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    # Forward pass through the model to get predictions
                    outputs = self(input_ids, attention_mask)
                    _, pred = torch.max(outputs, dim=1)
                    
                    # Extend predictions and actual lists with batch results
                    predictions.extend(pred.cpu().tolist())
                    actual.extend(labels.cpu().tolist())

                    # Update the progress bar to reflect batch processing
                    pbar.update(dataloader.batch_size)

        # Calculate accuracy based on predictions and actual labels and generate the classification report
        accuracy = accuracy_score(actual, predictions)
        report = classification_report(actual, predictions, target_names=labels_name)

        return accuracy, report

    def predict(self, dataloader: DataLoader) -> Any:
        """
        Make predictions using the trained Transformer model.

        :param `texts`: The list of input texts.
        :return: The predicted class labels.
        """
        # Set the model to evaluation mode (important for models with layers like dropout)
        self.eval()

        predictions = list()
        
        
        with torch.no_grad():
            with tqdm.tqdm(total = len(dataloader.dataset), desc = "PREDICTING:", colour = "cyan", ncols = 100) as pbar:
                for batch in dataloader:
                    
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = self(input_ids, attention_mask)
                    
                    _, pred = torch.max(outputs, dim=1)
                    predictions.extend(pred.cpu().tolist())
                    pbar.update(dataloader.batch_size)
        print("Prediction completed")
        return predictions
    
    def save_results(self, dataloader: DataLoader, path: str, labels_name: Optional[List[str]] = None) -> None:
        """
        Save evaluation results (accuracy and classification report) to a file.

        :param `dataloader`: The DataLoader providing the evaluation data.
        :param `path`: The path to save the results.
        :param `labels_name`: The names of the target labels (optional).
        """
        # Ensure the directory exists.
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Evaluate the model.
        accuracy, report = self.evaluate_model(dataloader, labels_name)

        # Save the results to the specified path.
        with open(path, 'a') as file:
            file.write(f"Models: {self.model} \n\n")
            file.write(f"Accuracy: {accuracy:.4f} \n\n")
            file.write(f"Classification Report: \n")
            file.write(f"{report} \n\n")

        response = f"| Report saved to {path} |"

        # Print a confirmation message.
        print("-" * len(response))
        print(response)
        print("-" * len(response))


    def train_val_curves(self, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, plot_epochs: int = 5, title: str = ""):
        """
        Learning and accuracy plot for the transformer
    
        :param `train_dataloader`: The training DataLoader
        :param `val_dataloader`: The validation DataLoader 
        :param `optimizer`: The optimizer for gradient descent.
        :param `scheduler`: The learning rate scheduler.
        :param `plot_epochs`: The number of epochs for the plots.
        :param `title`: The title of the plot
        """
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
    
        criterion = nn.CrossEntropyLoss()
    
        with tqdm.tqdm(total=(len(train_dataloader.dataset) + len(val_dataloader.dataset)) * plot_epochs, desc="PLOTTING:", colour="red") as pbar:
    
            for epoch in range(plot_epochs):
                # Training phase
                self.train()
                total_train_loss = 0
                correct_train_predictions = 0
                total_train_samples = 0
    
                for batch in train_dataloader:
                    optimizer.zero_grad()
    
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)
    
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
    
                    _, preds = torch.max(outputs, dim=1)
                    correct_train_predictions += torch.sum(preds == labels).item()
                    total_train_samples += labels.size(0)
                    total_train_loss += loss.item()
    
                    pbar.update(train_dataloader.batch_size)
    
                average_train_loss = total_train_loss / len(train_dataloader)
                train_losses.append(average_train_loss)
                train_accuracy = correct_train_predictions / total_train_samples
                train_accuracies.append(train_accuracy)
    
                # Validation phase
                self.eval()
                total_val_loss = 0
                correct_val_predictions = 0
                total_val_samples = 0
    
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["label"].to(self.device)
    
                        outputs = self(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
    
                        _, preds = torch.max(outputs, dim=1)
                        correct_val_predictions += torch.sum(preds == labels).item()
                        total_val_samples += labels.size(0)
                        total_val_loss += loss.item()
    
                        pbar.update(val_dataloader.batch_size)
    
                average_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(average_val_loss)
                val_accuracy = correct_val_predictions / total_val_samples
                val_accuracies.append(val_accuracy)
    
        # Drawing the plots
        epochs = range(1, plot_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, val_losses, label='Validation Loss', marker='o')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
    
        ax2.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
        ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
        fig.suptitle(title)
    
        # Mostriamo il grafico
        plt.show()

    

class Build_Handle:
    """
    A class to build dataset for transformer models.

    Attributes
    ----------
    `texts` : np.ndarray
        Array of input texts.
    `labels` : np.ndarray
        Array of labels corresponding to input texts.
    `tokenizer` : PreTrainedTokenizer
        Tokenizer to convert texts into token IDs.
    `max_length` : int, optional
        Maximum length of input sequences after tokenization.
    `cache` : dict
        Cache dictionary to store preprocessed data.

    """
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer, max_length=128) -> None:
        """
        Initialize the Build_Handle with texts, labels, tokenizer, and max_length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = dict()

    def __len__(self) -> int:
        """
        Return the length of the dataset (number of samples).
        """
        return len(self.texts)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a sample from the dataset by index, performing tokenization and caching.

        :param `index`: Index of the sample to retrieve.
        :return: Dictionary containing token IDs, attention mask, and label for the sample.
        """
        if index in self.cache:
            # If the data for the specified index is already cached, return it immediately.
            return self.cache[index]
        
        # Tokenize the text using the provided tokenizer.
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            return_tensors = "pt",          # Return PyTorch tensors
            max_length = self.max_length,   # Limit the maximum length of the tokenized sequence
            padding = "max_length",         # Pad sequences to the maximum length
            add_special_tokens = True,      # Add special tokens (like [CLS] and [SEP])
            truncation = True               # Truncate sequences longer than max_length
        )

        # Prepare the result dictionary containing tokenized input IDs, attention mask, and label.
        result = {
            "input_ids" : encoding["input_ids"].flatten(),          # Flatten tensor to a 1D array
            "attention_mask": encoding["attention_mask"].flatten(), # Flatten attention mask to a 1D array
            "label": torch.tensor(label)                            # Convert label to a PyTorch tensor
        }

        # Cache the result for future use.
        self.cache[index] = result

        return result