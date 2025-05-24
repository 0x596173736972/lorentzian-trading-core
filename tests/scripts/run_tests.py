"""
Script pour exécuter différents types de tests
"""

import sys
import subprocess
import argparse

def run_unit_tests():
    """Exécute les tests unitaires"""
    cmd = ["pytest", "tests/test_core/", "tests/test_features/", "tests/test_filters/", "-v"]
    return subprocess.run(cmd).returncode

def run_integration_tests():
    """Exécute les tests d'intégration"""
    cmd = ["pytest", "tests/test_integration/", "-v", "--timeout=60"]
    return subprocess.run(cmd).returncode

def run_performance_tests():
    """Exécute les tests de performance"""
    cmd = ["pytest", "tests/test_performance/", "-v", "--benchmark-only"]
    return subprocess.run(cmd).returncode

def run_all_tests():
    """Exécute tous les tests"""
    cmd = ["pytest", "tests/", "-v", "--cov=lorentzian_trading"]
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(description="Run tests for Lorentzian Trading Core")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "all"], 
                       default="all", help="Type of tests to run")
    
    args = parser.parse_args()
    
    if args.type == "unit":
        return run_unit_tests()
    elif args.type == "integration":
        return run_integration_tests()
    elif args.type == "performance":
        return run_performance_tests()
    else:
        return run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
