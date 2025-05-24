import pytest
import pandas as pd
import numpy as np
from lorentzian_trading.utils.data import DataProcessor

class TestDataProcessor:
    """Tests pour le processeur de données"""
    
    def test_clean_data_basic(self, sample_ohlcv_data):
        """Test du nettoyage basique"""
        processor = DataProcessor()
        
        # Ajouter quelques valeurs manquantes
        dirty_data = sample_ohlcv_data.copy()
        dirty_data.loc[dirty_data.index[10:15], 'close'] = np.nan
        
        clean_data = processor.clean_data(dirty_data)
        
        # Vérifier qu'il n'y a plus de NaN
        assert not clean_data.isna().any().any()
        assert len(clean_data) <= len(dirty_data)
    
    def test_validate_ohlcv(self, sample_ohlcv_data):
        """Test de validation OHLCV"""
        processor = DataProcessor()
        
        # Données valides
        assert processor.validate_ohlcv(sample_ohlcv_data) is True
        
        # Données invalides (colonnes manquantes)
        invalid_data = sample_ohlcv_data.drop('close', axis=1)
        assert processor.validate_ohlcv(invalid_data) is False
