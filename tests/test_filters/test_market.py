import pytest
import pandas as pd
import numpy as np
from lorentzian_trading.filters.market import MarketFilters, FilterConfig

class TestMarketFilters:
    """Tests pour la classe MarketFilters"""
    
    def test_init_default(self):
        """Test d'initialisation par défaut"""
        filters = MarketFilters()
        assert filters.config.use_volatility_filter is True
        assert filters.config.use_regime_filter is True
    
    def test_init_custom_config(self):
        """Test avec configuration personnalisée"""
        config = FilterConfig(
            use_volatility_filter=False,
            regime_threshold=-0.2
        )
        filters = MarketFilters(config)
        
        assert filters.config.use_volatility_filter is False
        assert filters.config.regime_threshold == -0.2
    
    def test_apply_all_filters(self, sample_ohlcv_data):
        """Test d'application de tous les filtres"""
        filters = MarketFilters()
        result = filters.apply_all_filters(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.dtype == bool
    
    def test_volatility_filter(self, sample_ohlcv_data):
        """Test du filtre de volatilité"""
        filters = MarketFilters()
        result = filters.volatility_filter(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        # Il doit y avoir des périodes filtrées (False) et acceptées (True)
        assert result.sum() > 0
        assert not result.all()
    
    def test_regime_filter(self, sample_ohlcv_data):
        """Test du filtre de régime"""
        filters = MarketFilters()
        result = filters.regime_filter(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
    
    def test_filters_with_disabled_config(self, sample_ohlcv_data):
        """Test avec filtres désactivés"""
        config = FilterConfig(
            use_volatility_filter=False,
            use_regime_filter=False,
            use_adx_filter=False
        )
        filters = MarketFilters(config)
        result = filters.apply_all_filters(sample_ohlcv_data)
        
        # Tous les filtres désactivés = tous True
        assert result.all()
