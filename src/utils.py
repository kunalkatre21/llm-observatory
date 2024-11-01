import logging
import json
import requests
from datetime import datetime
import pandas as pd
from typing import Dict, Any
from config import Settings

settings = Settings()  # Initialize settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_testing.log'),
        logging.StreamHandler()
    ]
)

def call_ai(prompt: str, model: str) -> Dict[str, Any]:
    """
    Call AI model with the given prompt and return response with metrics
    """
    start_time = datetime.now()

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}"
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=settings.timeout
        )
        response.raise_for_status()
        data = response.json()

        return {
            'success': True,
            'response': data['choices'][0]['message']['content'],
            'duration': (datetime.now() - start_time).total_seconds(),
            'model': model,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error calling AI: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'duration': (datetime.now() - start_time).total_seconds(),
            'model': model,
            'timestamp': datetime.now().isoformat()
        }

def export_results(results, format="csv"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if format == "csv":
        df = pd.DataFrame(results)
        df.to_csv(f"results_{now}.csv", index=False)
    elif format == "json":
        with open(f"results_{now}.json", "w") as f:
            json.dump(results, f)
