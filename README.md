# 🚀 Lorentzian Trading Core

A quantum-inspired machine learning library for financial trading using Lorentzian distance metrics.

## 🌟 Features

- **Lorentzian Distance**: Advanced distance metric that accounts for market warping effects
- **ML Classification**: Specialized classifier for predicting price direction
- **Feature Engineering**: Comprehensive technical indicator generation
- **Market Filters**: Smart filtering system for market conditions
- **Backtesting**: Built-in backtesting and performance analysis
- **Kernel Regression**: Nadaraya-Watson kernel regression support

## 📦 Installation

```bash
pip install lorentzian-trading-core
```

## 🚀 Quick Start

```python
import pandas as pd
from lorentzian_trading import LorentzianClassifier, FeatureEngine

# Load your data
data = pd.read_csv('your_data.csv')  # OHLCV format

# Initialize classifier
classifier = LorentzianClassifier()

# Fit the model
features = ['RSI', 'MACD', 'ADX', 'CCI']
classifier.fit(data, features)

# Make predictions
current_features = [65.2, 0.8, 25.3, -45.1]  # Current feature values
prediction = classifier.predict(current_features)

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 🔬 Research Background

This library implements the concepts from the research on Lorentzian distance in financial markets, 
addressing the limitations of traditional Euclidean distance in the presence of market warping effects 
caused by significant economic events.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
