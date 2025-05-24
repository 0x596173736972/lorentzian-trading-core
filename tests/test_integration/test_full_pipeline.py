import pytest
import pandas as pd
import numpy as np
from lorentzian_trading import (
    LorentzianClassifier, 
    FeatureEngine, 
    MarketFilters,
    DataProcessor
)

class TestFullPipeline:
    """Tests d'intégration du pipeline complet"""
    
    def test_complete_workflow(self, sample_ohlcv_data):
        """Test du workflow complet"""
        # 1. Préparation des données
        processor = DataProcessor()
        clean_data = processor.clean_data(sample_ohlcv_data)
        
        # 2. Génération des features
        engine = FeatureEngine()
        features = ['RSI', 'CCI', 'ADX']
        feature_data = engine.generate_features(clean_data, features)
        
        # 3. Application des filtres
        filters = MarketFilters()
        market_filter = filters.apply_all_filters(clean_data)
        
        # 4. Entraînement du classificateur
        classifier = LorentzianClassifier()
        classifier.fit(clean_data, features)
        
        # 5. Prédictions
        current_features = feature_data.iloc[-1].values
        prediction = classifier.predict(current_features)
        
        # Vérifications
        assert prediction['prediction'] in [-1, 0, 1]
        assert 0 <= prediction['confidence'] <= 1
        assert prediction['neighbors_used'] > 0
    
    def test_backtesting_simulation(self, sample_ohlcv_data):
        """Simulation de backtesting"""
        classifier = LorentzianClassifier()
        engine = FeatureEngine()
        features = ['RSI', 'MACD']
        
        # Entraînement sur les 70% premiers
        train_size = int(len(sample_ohlcv_data) * 0.7)
        train_data = sample_ohlcv_data.iloc[:train_size]
        test_data = sample_ohlcv_data.iloc[train_size:]
        
        classifier.fit(train_data, features)
        
        # Test sur les 30% restants
        feature_data = engine.generate_features(sample_ohlcv_data, features)
        predictions = []
        
        for i in range(train_size, len(sample_ohlcv_data) - 1):
            current_features = feature_data.iloc[i].values
            pred = classifier.predict(current_features)
            predictions.append(pred['prediction'])
        
        # Vérifications
        assert len(predictions) > 0
        assert all(p in [-1, 0, 1] for p in predictions)
    
    def test_memory_usage(self, sample_ohlcv_data):
        """Test d'utilisation mémoire"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Traitement intensif
        classifier = LorentzianClassifier()
        engine = FeatureEngine()
        
        for _ in range(10):
            features = engine.generate_features(sample_ohlcv_data, ['RSI', 'MACD', 'CCI'])
            classifier.fit(sample_ohlcv_data, ['RSI'])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # L'augmentation mémoire ne doit pas être excessive
        assert memory_increase < 100  # Moins de 100MB
