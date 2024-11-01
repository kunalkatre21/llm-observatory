from dataclasses import dataclass
import psutil
import streamlit as st  # Add the missing import for streamlit

@dataclass
class PerformanceMetrics:
    tokens_per_second: float
    memory_used_mb: float
    cost_estimate: float
    success_rate: float

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "Performance": {
                "token_speed": [],
                "response_times": [],
                "memory_usage": []
            },
            "Quality": {
                "success_rates": {},
                "failure_patterns": {},
                "consistency_scores": {}
            }
        }

    def calculate_token_speed(self, response_text, time_taken):
        estimated_tokens = len(response_text.split()) * 1.3
        return estimated_tokens / time_taken

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB

class TestProgress:
    def __init__(self):
        self.progress_bar = None
        self.status_text = None

    def init_progress(self, total_steps):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def update(self, step, message):
        if self.progress_bar:
            self.progress_bar.progress(step)
        if self.status_text:
            self.status_text.text(message)
