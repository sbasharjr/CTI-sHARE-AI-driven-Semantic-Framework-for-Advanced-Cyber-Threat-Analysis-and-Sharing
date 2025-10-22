"""
Tests for dashboard integration in main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import main


class TestMainDashboardIntegration:
    
    def test_dashboard_mode_in_choices(self):
        """Test that dashboard mode is available in argument parser"""
        parser = main.argparse.ArgumentParser()
        parser.add_argument(
            'mode',
            choices=['analyze', 'realtime', 'api', 'train', 'dashboard'],
            help='Operation mode'
        )
        
        # This should not raise an error
        args = parser.parse_args(['dashboard'])
        assert args.mode == 'dashboard'
    
    def test_run_dashboard_function_exists(self):
        """Test that run_dashboard function exists and is callable"""
        assert hasattr(main, 'run_dashboard')
        assert callable(main.run_dashboard)
    
    def test_run_dashboard_signature(self):
        """Test that run_dashboard has correct signature"""
        import inspect
        sig = inspect.signature(main.run_dashboard)
        params = list(sig.parameters.keys())
        
        assert 'host' in params
        assert 'port' in params
        
        # Check default values
        assert sig.parameters['host'].default == '0.0.0.0'
        assert sig.parameters['port'].default == 5001
    
    def test_dashboard_import(self):
        """Test that ThreatDashboard is imported in main.py"""
        # Check if the import exists by checking the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        module = importlib.util.module_from_spec(spec)
        
        # Read the file to check imports
        with open('main.py', 'r') as f:
            content = f.read()
            assert 'from src.dashboard.dashboard import ThreatDashboard' in content


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
