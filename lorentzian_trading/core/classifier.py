import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from .distance import LorentzianDistance
from ..features.engineering import FeatureEngine
from ..filters.market import MarketFilters

@dataclass
class ClassifierConfig:
    """Configuration for Lorentzian Classifier"""
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False
    use_dynamic_exits: bool = False

class LorentzianClassifier:
    """
    Machine Learning classifier using Lorentzian distance for financial time series prediction.
    
    The classifier specializes in predicting price direction over the next 4 bars by using
    an Approximate Nearest Neighbors algorithm with Lorentzian distance to account for
    the warping effects of significant economic events.
    
    Attributes:
        config: Configuration object with classifier parameters
        feature_engine: Feature engineering component
        distance_calculator: Lorentzian distance calculator
        market_filters: Market condition filters
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.feature_engine = FeatureEngine()
        self.distance_calculator = LorentzianDistance()
        self.market_filters = MarketFilters()
        
        # Internal state
        self._training_labels: List[int] = []
        self._feature_arrays: Dict[str, List[float]] = {}
        self._predictions: List[float] = []
        self._distances: List[float] = []
        self._last_distance: float = -1.0
        
    def fit(self, data: pd.DataFrame, features: List[str]) -> 'LorentzianClassifier':
        """
        Fit the classifier on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            features: List of feature names to use
            
        Returns:
            Self for method chaining
        """
        # Generate features
        feature_data = self.feature_engine.generate_features(data, features)
        
        # Generate training labels (4-bar future direction)
        labels = self._generate_training_labels(data['close'])
        
        # Store training data
        self._training_labels = labels
        self._feature_arrays = {
            f'f{i+1}': feature_data.iloc[:, i].tolist() 
            for i in range(len(features))
        }
        
        return self
    
    def predict(self, current_features: np.ndarray) -> Dict[str, Union[int, float]]:
        """
        Predict the direction of price movement for the next 4 bars.
        
        Args:
            current_features: Array of current feature values
            
        Returns:
            Dictionary containing prediction and confidence metrics
        """
        if not self._training_labels:
            raise ValueError("Classifier must be fitted before making predictions")
        
        # Reset prediction arrays
        predictions = []
        distances = []
        last_distance = -1.0
        
        # Approximate Nearest Neighbors with Lorentzian Distance
        size = min(self.config.max_bars_back - 1, len(self._training_labels) - 1)
        
        for i in range(0, size, 4):  # Step by 4 for chronological spacing
            distance = self.distance_calculator.calculate(
                current_features, 
                self._get_historical_features(i),
                self.config.feature_count
            )
            
            if distance >= last_distance:
                last_distance = distance
                distances.append(distance)
                predictions.append(self._training_labels[i])
                
                # Maintain k-nearest neighbors
                if len(predictions) > self.config.neighbors_count:
                    # Use distance in lower 25% as threshold
                    threshold_idx = int(self.config.neighbors_count * 0.75)
                    last_distance = sorted(distances)[threshold_idx]
                    distances.pop(0)
                    predictions.pop(0)
        
        # Final prediction
        raw_prediction = sum(predictions) if predictions else 0
        
        return {
            'prediction': np.sign(raw_prediction),
            'confidence': abs(raw_prediction) / len(predictions) if predictions else 0,
            'raw_score': raw_prediction,
            'neighbors_used': len(predictions)
        }
    
    def _generate_training_labels(self, prices: pd.Series) -> List[int]:
        """Generate training labels based on 4-bar future direction"""
        labels = []
        for i in range(len(prices) - 4):
            current_price = prices.iloc[i]
            future_price = prices.iloc[i + 4]
            
            if future_price > current_price:
                labels.append(1)  # Long
            elif future_price < current_price:
                labels.append(-1)  # Short
            else:
                labels.append(0)  # Neutral
                
        return labels
    
    def _get_historical_features(self, index: int) -> np.ndarray:
        """Get historical feature values at given index"""
        features = []
        for i in range(self.config.feature_count):
            feature_key = f'f{i+1}'
            if feature_key in self._feature_arrays:
                features.append(self._feature_arrays[feature_key][index])
            else:
                features.append(0.0)
        return np.array(features)
