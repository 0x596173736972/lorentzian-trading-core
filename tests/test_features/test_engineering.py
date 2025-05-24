import pytest
import pandas as pd
import numpy as np
from lorentzian_trading.features.engineering import FeatureEngine

class TestFeatureEngine:
    """Tests pour la classe FeatureEngine"""
    
    def test_init(self):
        """Test d'initialisation"""
        engine = FeatureEngine()
        assert len(engine.feature_functions) > 0
        assert 'RSI' in engine.feature_functions
        assert 'MACD' in engine.feature_functions
    
    def test_generate_features_single(self, sample_ohlcv_data):
        """Test de génération d'une seule feature"""
        engine = FeatureEngine()
        features = engine.generate_features(sample_ohlcv_data, ['RSI'])
        
        assert isinstance(features, pd.DataFrame)
        assert 'RSI' in features.columns
        assert len(features) == len(sample_ohlcv_data)
        assert not features['RSI'].isna().all()
    
    def test_generate_features_multiple(self, sample_ohlcv_data):
        """Test de génération de multiples features"""
        engine = FeatureEngine()
        feature_list = ['RSI', 'CCI', 'ADX']
        features = engine.generate_features(sample_ohlcv_data, feature_list)
        
        assert isinstance(features, pd.DataFrame)
        for feature in feature_list:
            assert feature in features.columns
        
        # Vérification que les données ne sont pas toutes NaN
        for col in features.columns:
            assert not features[col].isna().all()
    
    def test_generate_features_with_params(self, sample_ohlcv_data):
        """Test avec paramètres personnalisés"""
        engine = FeatureEngine()
        params = {'RSI': {'period': 21}}
        features = engine.generate_features(sample_ohlcv_data, ['RSI'], params)
        
        assert 'RSI' in features.columns
        # Le RSI doit être calculé correctement
        assert features['RSI'].max() <= 100
        assert features['RSI'].min() >= 0
    
    def test_generate_features_unknown(self, sample_ohlcv_data):
        """Test avec feature inconnue"""
        engine = FeatureEngine()
        
        with pytest.raises(ValueError, match="Unknown feature"):
            engine.generate_features(sample_ohlcv_data, ['UNKNOWN_FEATURE'])
    
    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test spécifique du calcul RSI"""
        engine = FeatureEngine()
        rsi = engine._rsi(sample_ohlcv_data, period=14)
        
        # RSI doit être entre 0 et 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(0 <= val <= 100 for val in valid_rsi)
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Test spécifique du calcul MACD"""
        engine = FeatureEngine()
        macd_data = engine._macd(sample_ohlcv_data)
        
        assert isinstance(macd_data, pd.DataFrame)
        assert 'macd' in macd_data.columns
        assert 'signal' in macd_data.columns
        assert 'histogram' in macd_data.columns
