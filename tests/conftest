import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

@pytest.fixture
def sample_ohlcv_data():
    """Génère des données OHLCV de test"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Simulation d'un prix avec tendance et volatilité
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n)  # Rendements quotidiens
    prices = [base_price]
    
    for i in range(1, n):
        prices.append(prices[-1] * (1 + returns[i]))
    
    # Génération OHLCV réaliste
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Volatilité intraday
        intraday_vol = abs(np.random.normal(0, 0.01))
        high = close * (1 + intraday_vol)
        low = close * (1 - intraday_vol)
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def feature_data():
    """Données de features pour les tests"""
    return np.array([65.2, 0.8, 25.3, -45.1, 12.5])

@pytest.fixture
def temp_dir():
    """Répertoire temporaire pour les tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
