import pytest
import numpy as np
from lorentzian_trading.core.distance import LorentzianDistance

class TestLorentzianDistance:
    """Tests pour la classe LorentzianDistance"""
    
    def test_calculate_basic(self):
        """Test du calcul de distance basique"""
        point1 = [1, 2, 3]
        point2 = [4, 5, 6]
        
        distance = LorentzianDistance.calculate(point1, point2)
        
        # Distance Lorentzienne: log(1+|1-4|) + log(1+|2-5|) + log(1+|3-6|)
        expected = np.log(1 + 3) + np.log(1 + 3) + np.log(1 + 3)
        assert abs(distance - expected) < 1e-10
    
    def test_calculate_with_feature_count(self):
        """Test avec limitation du nombre de features"""
        point1 = [1, 2, 3, 4, 5]
        point2 = [2, 3, 4, 5, 6]
        
        distance = LorentzianDistance.calculate(point1, point2, feature_count=3)
        
        # Seules les 3 premières features doivent être utilisées
        expected = 3 * np.log(1 + 1)  # |1-2| = |2-3| = |3-4| = 1
        assert abs(distance - expected) < 1e-10
    
    def test_calculate_zero_distance(self):
        """Test avec points identiques"""
        point1 = [1, 2, 3]
        point2 = [1, 2, 3]
        
        distance = LorentzianDistance.calculate(point1, point2)
        
        # log(1 + 0) = 0
        assert distance == 0
    
    def test_calculate_numpy_arrays(self):
        """Test avec des arrays NumPy"""
        point1 = np.array([1.5, 2.7, 3.2])
        point2 = np.array([4.1, 5.3, 6.8])
        
        distance = LorentzianDistance.calculate(point1, point2)
        
        expected = (np.log(1 + abs(1.5 - 4.1)) + 
                   np.log(1 + abs(2.7 - 5.3)) + 
                   np.log(1 + abs(3.2 - 6.8)))
        
        assert abs(distance - expected) < 1e-10
    
    def test_euclidean_distance(self):
        """Test de la distance euclidienne"""
        point1 = [1, 2, 3]
        point2 = [4, 5, 6]
        
        distance = LorentzianDistance.euclidean_distance(point1, point2)
        expected = np.sqrt(9 + 9 + 9)  # sqrt(3² + 3² + 3²)
        
        assert abs(distance - expected) < 1e-10
    
    def test_manhattan_distance(self):
        """Test de la distance de Manhattan"""
        point1 = [1, 2, 3]
        point2 = [4, 5, 6]
        
        distance = LorentzianDistance.manhattan_distance(point1, point2)
        expected = 3 + 3 + 3  # |1-4| + |2-5| + |3-6|
        
        assert distance == expected
