import pytest
from src.test_runner import AdvancedTestRunner
from src.metrics import MetricsCollector

def test_test_runner_initialization():
    runner = AdvancedTestRunner()
    assert hasattr(runner, 'test_patterns')
    assert 'chain_of_thought' in runner.test_patterns

def test_metrics_collector_initialization():
    collector = MetricsCollector()
    assert 'Performance' in collector.metrics
    assert 'Quality' in collector.metrics