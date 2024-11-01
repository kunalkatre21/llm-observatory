import logging
import time
from config import Settings

settings = Settings()  # Initialize settings

class AITestingError(Exception):
    def __init__(self, message, error_type="general"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

def handle_api_error(error, retry_count=0):
    if retry_count < settings.max_retries:
        logging.warning(f"Retrying... Attempt {retry_count + 1}")
        time.sleep(2 ** retry_count)  # Exponential backoff
        return True
    return False
