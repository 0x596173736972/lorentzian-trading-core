import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
import talib

class FeatureEngine:
    """
    Feature engineering component for generating technical indicators
    used as input features for the Lorentzian classifier.
    """
    
    def __init__(self):
        self.feature_functions: Dict[str, Callable] = {
            'RSI': self._rsi,
            'WT': self._williams_transform,
            'CCI': self._cci,
            'ADX': self._adx,
            'MACD': self._macd,
            'BB': self._bollinger_bands,
            'STOCH': self._stochastic,
            'EMA': self._ema,
            'SMA': self._sma,
        }
    
    def generate_features(
        self, 
        data: pd.DataFrame, 
        feature_list: List[str],
        params: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        Generate technical indicator features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            feature_list: List of feature names to generate
            params: Optional parameters for each feature
            
        Returns:
            DataFrame with generated features
        """
        if params is None:
            params = {}
            
        features = pd.DataFrame(index=data.index)
        
        for feature_name in feature_list:
            if feature_name in self.feature_functions:
                feature_params = params.get(feature_name, {})
                feature_values = self.feature_functions[feature_name](data, **feature_params)
                
                if isinstance(feature_values, pd.DataFrame):
                    # Multiple columns returned
                    for col in feature_values.columns:
                        features[f"{feature_name}_{col}"] = feature_values[col]
                else:
                    # Single column returned
                    features[feature_name] = feature_values
            else:
                raise ValueError(f"Unknown feature: {feature_name}")
        
        return features.fillna(method='ffill').fillna(0)
    
    def _rsi(self, data: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """Relative Strength Index"""
        return talib.RSI(data[price_col].values, timeperiod=period)
    
    def _williams_transform(self, data: pd.DataFrame, period1: int = 10, period2: int = 11) -> pd.Series:
        """Williams Transform (simplified version)"""
        hlc3 = (data['high'] + data['low'] + data['close']) / 3
        esa = hlc3.ewm(span=period1).mean()
        d = np.abs(hlc3 - esa).ewm(span=period1).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        wt1 = ci.ewm(span=period2).mean()
        return wt1
    
    def _cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        return talib.CCI(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    def _adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        return talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=period)
    
    def _macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD Indicator"""
        macd, signal_line, histogram = talib.MACD(
            data['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)
    
    def _bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            data['close'].values, timeperiod=period, nbdevup=std, nbdevdn=std
        )
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': (upper - lower) / middle,
            'percent_b': (data['close'] - lower) / (upper - lower)
        }, index=data.index)
    
    def _stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        k_percent, d_percent = talib.STOCH(
            data['high'].values, data['low'].values, data['close'].values,
            fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
        )
        return pd.DataFrame({
            'k': k_percent,
            'd': d_percent
        }, index=data.index)
    
    def _ema(self, data: pd.DataFrame, period: int = 20, price_col: str = 'close') -> pd.Series:
        """Exponential Moving Average"""
        return talib.EMA(data[price_col].values, timeperiod=period)
    
    def _sma(self, data: pd.DataFrame, period: int = 20, price_col: str = 'close') -> pd.Series:
        """Simple Moving Average"""
        return talib.SMA(data[price_col].values, timeperiod=period)

# ==============================================================================
# lorentzian_trading/filters/market.py
# ==============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class FilterConfig:
    """Configuration for market filters"""
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20
    use_ema_filter: bool = False
    ema_period: int = 200
    use_sma_filter: bool = False
    sma_period: int = 200

class MarketFilters:
    """
    Market condition filters for improving ML prediction accuracy.
    
    These filters help to avoid trading during unfavorable market conditions
    such as high volatility, ranging markets, or adverse trends.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
    
    def apply_all_filters(self, data: pd.DataFrame) -> pd.Series:
        """
        Apply all enabled filters and return combined filter result.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Boolean Series indicating when all filters pass
        """
        filters = pd.Series(True, index=data.index)
        
        if self.config.use_volatility_filter:
            filters &= self.volatility_filter(data)
        
        if self.config.use_regime_filter:
            filters &= self.regime_filter(data)
            
        if self.config.use_adx_filter:
            filters &= self.adx_filter(data)
            
        if self.config.use_ema_filter:
            filters &= self.ema_filter(data)
            
        if self.config.use_sma_filter:
            filters &= self.sma_filter(data)
        
        return filters
    
    def volatility_filter(self, data: pd.DataFrame, min_periods: int = 1, max_periods: int = 10) -> pd.Series:
        """
        Filter based on market volatility.
        
        Avoids trading during extremely volatile periods that may produce
        unreliable signals.
        """
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=max_periods).std()
        
        # Filter out extreme volatility (top and bottom percentiles)
        vol_threshold_high = volatility.quantile(0.95)
        vol_threshold_low = volatility.quantile(0.05)
        
        return (volatility >= vol_threshold_low) & (volatility <= vol_threshold_high)
    
    def regime_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Filter based on market regime (trending vs ranging).
        
        Uses a trend detection mechanism to identify favorable trending conditions.
        """
        # Calculate price momentum
        ohlc4 = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        momentum = ohlc4.pct_change(10)  # 10-period momentum
        
        # Regime filter based on momentum threshold
        return momentum > self.config.regime_threshold
    
    def adx_filter(self, data: pd.DataFrame) -> pd.Series:
        """
        Filter based on Average Directional Index.
        
        ADX measures trend strength; higher values indicate stronger trends.
        """
        try:
            import talib
            adx = talib.ADX(
                data['high'].values, 
                data['low'].values, 
                data['close'].values, 
                timeperiod=14
            )
            return pd.Series(adx, index=data.index) > self.config.adx_threshold
        except ImportError:
            # Fallback implementation
            return pd.Series(True, index=data.index)
    
    def ema_filter(self, data: pd.DataFrame) -> pd.Series:
        """Filter based on EMA trend direction"""
        ema = data['close'].ewm(span=self.config.ema_period).mean()
        return data['close'] > ema  # Bullish when price above EMA
    
    def sma_filter(self, data: pd.DataFrame) -> pd.Series:
        """Filter based on SMA trend direction"""
        sma = data['close'].rolling(window=self.config.sma_period).mean()
        return data['close'] > sma  # Bullish when price above SMA
