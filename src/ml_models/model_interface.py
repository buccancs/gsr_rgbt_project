# src/ml_models/model_interface.py

"""
Model interface module for the ML pipeline.

This module defines abstract base classes for models and model factories,
providing a common interface for different model implementations (PyTorch, etc.).
This makes the pipeline more modular and allows for easy swapping of model implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List

import numpy as np

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    This class defines the common interface that all model implementations must follow,
    regardless of the underlying framework (PyTorch, etc.).
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            **kwargs: Additional arguments for training
            
        Returns:
            Dict[str, Any]: Training history or metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path (str): Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            BaseModel: Loaded model
        """
        pass


class ModelFactory(ABC):
    """
    Abstract factory for creating models.
    
    This class defines the interface for model factories, which are responsible
    for creating model instances based on configuration parameters.
    """
    
    @abstractmethod
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on the given configuration.
        
        Args:
            input_shape (Tuple): Shape of the input data
            config (Dict[str, Any]): Model configuration parameters
            
        Returns:
            BaseModel: A model instance
        """
        pass


class ModelRegistry:
    """
    Registry for model factories.
    
    This class maintains a registry of model factories, allowing models to be
    created by name without knowing the specific implementation details.
    """
    
    _factories = {}
    
    @classmethod
    def register(cls, name: str, factory: ModelFactory) -> None:
        """
        Register a model factory.
        
        Args:
            name (str): Name to register the factory under
            factory (ModelFactory): Factory instance to register
        """
        cls._factories[name] = factory
        logging.info(f"Registered model factory: {name}")
    
    @classmethod
    def create_model(cls, name: str, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model using the registered factory.
        
        Args:
            name (str): Name of the factory to use
            input_shape (Tuple): Shape of the input data
            config (Dict[str, Any]): Model configuration parameters
            
        Returns:
            BaseModel: A model instance
            
        Raises:
            ValueError: If no factory is registered with the given name
        """
        if name not in cls._factories:
            raise ValueError(f"No model factory registered with name: {name}")
        
        return cls._factories[name].create_model(input_shape, config)
    
    @classmethod
    def get_registered_models(cls) -> List[str]:
        """
        Get a list of all registered model names.
        
        Returns:
            List[str]: List of registered model names
        """
        return list(cls._factories.keys())