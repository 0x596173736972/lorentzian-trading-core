import pytest
import time
import pandas as pd
import numpy as np
from lorentzian_trading.core.distance import LorentzianDistance
from lorentzian_trading.core.classifier import LorentzianClassifier

class TestPerformanceBenchmarks:
    """Tests de performance et benchmarks"""
    
    def test_distance_calculation_speed(self):
        """Benchmark du calcul de distance"""
        np.random.seed(42)
        point1 = np.random.random(100)
        point2 = np.random.random(100)
        
        start_time = time.time()
        for _ in range(1000):
            LorentzianDistance.calculate(point1, point2)
        end_time = time.time()
        
        # Doit calculer 1000 distances en moins d'1 seconde
        assert (end_time - start_time) < 1.0
    
    def test_classifier_training_speed(self, sample_ohlcv_data):
        """Benchmark de l'entraînement"""
        classifier = LorentzianClassifier()
        
        start_time = time.time()
        classifier.fit(sample_ohlcv_data, ['RSI'])
        end_time = time.time()
        
        # L'entraînement doit être rapide
        assert (end_time - start_time) < 5.0
    
    def test_prediction_speed(self, sample_ohlcv_data):
        """Benchmark des prédictions"""
        classifier = LorentzianClassifier()
        classifier.fit(sample_ohlcv_data, ['RSI'])
        
        features = np.array([65.2])
        
        start_time = time.time()
        for _ in range(100):
            classifier.predict(features)
        end_time = time.time()
        
        # 100 prédictions en moins de 1 seconde
        assert (end_time - start_time) < 1.0
