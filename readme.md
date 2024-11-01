# LLM Observatory 🔭

A visual dashboard for testing and monitoring Large Language Models (LLMs), born from a UX designer's perspective on AI testing.

## 🌟 Key Features

- **Visual Test Runner**: Interactive dashboard to run tests on different LLM models
- **Multiple Testing Modes**: 
  - Full Test Suite
  - Quick Test
  - Custom Test Builder
- **Performance Monitoring**: Real-time metrics and historical performance tracking
- **Test Categories**:
  - Chain of Thought
  - Consistency Checks
  - Hallucination Detection
  - Prompt Injection Tests

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/kunalkatre21/llm-observatory.git
cd llm-observatory
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create .env file with:
OPENROUTER_API_KEY=your_api_key_here
```

5. Run the application
```bash
streamlit run src/app.py
```

## 📊 Dashboard Overview

[Add a screenshot of your dashboard here]

## 🎯 Test Types Explained

### Chain of Thought Tests
Evaluates the model's ability to show step-by-step reasoning through problems.

### Consistency Tests
Checks if the model provides consistent answers across multiple identical queries.

### Hallucination Checks
Monitors for false or made-up information in model responses.

### Prompt Injection Tests
Ensures model maintains appropriate boundaries and security.

## 🛠 Project Structure
```
llm-observatory/
├── src/
│   ├── app.py            # Main Streamlit application
│   ├── test_runner.py    # Test execution logic
│   ├── metrics.py        # Performance metrics collection
│   ├── visualizer.py     # Data visualization components
│   ├── utils.py          # Utility functions
│   └── config.py         # Configuration management
└── docs/                 # Additional documentation
```

## 🎨 Design Philosophy

As a UX designer venturing into AI testing, this project emphasizes:
- Visual feedback and intuitive test monitoring
- Clear presentation of complex test results
- Accessible technical information for non-technical users

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests for improvements
- Share your testing templates

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.