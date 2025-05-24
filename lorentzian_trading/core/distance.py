import numpy as np
from typing import Union

class LorentzianDistance:
    """
    Lorentzian distance calculator for financial time series.
    
    Lorentzian distance is used as an alternative to Euclidean distance to better
    account for the warping effects of significant economic events on price-time
    relationships in financial markets.
    """
    
    @staticmethod
    def calculate(
        point1: Union[np.ndarray, list], 
        point2: Union[np.ndarray, list],
        feature_count: Optional[int] = None
    ) -> float:
        """
        Calculate Lorentzian distance between two feature vectors.
        
        The Lorentzian distance is computed as:
        d = Σ log(1 + |x_i - y_i|) for i in range(feature_count)
        
        Args:
            point1: First feature vector
            point2: Second feature vector  
            feature_count: Number of features to use (optional)
            
        Returns:
            Lorentzian distance as float
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        if feature_count is not None:
            p1 = p1[:feature_count]
            p2 = p2[:feature_count]
        
        # Lorentzian distance formula
        differences = np.abs(p1 - p2)
        log_terms = np.log(1 + differences)
        return np.sum(log_terms)
    
    @staticmethod
    def euclidean_distance(
        point1: Union[np.ndarray, list], 
        point2: Union[np.ndarray, list]
    ) -> float:
        """Calculate Euclidean distance for comparison purposes"""
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    @staticmethod
    def manhattan_distance(
        point1: Union[np.ndarray, list], 
        point2: Union[np.ndarray, list]
    ) -> float:
        """Calculate Manhattan distance for comparison purposes"""
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.sum(np.abs(p1 - p2))
