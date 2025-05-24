import pytest
import pandas as pd
import numpy as np
from lorentzian_trading.core.classifier import LorentzianClassifier, ClassifierConfig

class TestLorentzianClassifier:
    """Tests pour la classe LorentzianClassifier"""
    
    def test_init_default_config(self):
        """Test d'initialisation avec config par défaut"""
        classifier = LorentzianClassifier()
        
        assert classifier.config.neighbors_count == 8
        assert classifier.config.max_bars_back == 2000
        assert classifier.config.feature_count == 5
        assert classifier.feature_engine is not None
        assert classifier.distance_calculator is not None
    
    def test_init_custom_config(self):
        """Test d'initialisation avec config personnalisée"""
        config = ClassifierConfig(
            neighbors_count=10,
            max_bars_back=1000,
            feature_count=3
        )
        classifier = LorentzianClassifier(config)
        
        assert classifier.config.neighbors_count == 10
        assert classifier.config.max_bars_back == 1000
        assert classifier.config.feature_count == 3
    
    def test_fit(self, sample_ohlcv_data):
        """Test de l'entraînement du classificateur"""
        classifier = LorentzianClassifier()
        features = ['RSI', 'MACD']
        
        # Doit passer sans erreur
        result = classifier.fit(sample_ohlcv_data, features)
        
        assert result is classifier  # Retourne self
        assert len(classifier._training_labels) > 0
        assert len(classifier._feature_arrays) > 0
    
    def test_predict_without_fit(self):
        """Test de prédiction sans entraînement"""
        classifier = LorentzianClassifier()
        features = np.array([65.2, 0.8, 25.3])
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict(features)
    
    def test_predict_with_fit(self, sample_ohlcv_data):
        """Test de prédiction après entraînement"""
        classifier = LorentzianClassifier()
        features = ['RSI']
        
        classifier.fit(sample_ohlcv_data, features)
        
        current_features = np.array([65.2])
        prediction = classifier.predict(current_features)
        
        assert 'prediction' in prediction
        assert 'confidence' in prediction
        assert 'raw_score' in prediction
        assert 'neighbors_used' in prediction
        
        assert prediction['prediction'] in [-1, 0, 1]
        assert 0 <= prediction['confidence'] <= 1
        assert isinstance(prediction['neighbors_used'], int)
    
    def test_generate_training_labels(self, sample_ohlcv_data):
        """Test de génération des labels d'entraînement"""
        classifier = LorentzianClassifier()
        prices = sample_ohlcv_data['close']
        
        labels = classifier._generate_training_labels(prices)
        
        # Doit avoir 4 labels de moins que le nombre de prix
        assert len(labels) == len(prices) - 4
        
        # Tous les labels doivent être -1, 0, ou 1
        for label in labels:
            assert label in [-1, 0, 1]
