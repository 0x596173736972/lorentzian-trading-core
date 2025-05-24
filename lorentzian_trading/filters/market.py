
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
