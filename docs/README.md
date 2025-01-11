# FinGPT Trader Documentation

## Table of Contents

1. [Overview](overview.md)
2. [Architecture](architecture/README.md)
3. [API Reference](api/README.md)
4. [Models](models/README.md)
5. [Services](services/README.md)
6. [Deployment](deployment/README.md)
7. [Development](development/README.md)

## Quick Start

1. **Installation**
```bash
git clone https://github.com/ashioyajotham/fingpt_trader.git
cd fingpt_trader
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. **Configuration**
- Copy `config/example.yaml` to `config/trading.yaml`
- Set up your environment variables:
  ```bash
  export HUGGINGFACE_TOKEN="your_token"
  export BINANCE_API_KEY="your_key"
  export BINANCE_API_SECRET="your_secret"
  ```

3. **Run Tests**
```bash
pytest tests/
```

4. **Start Trading**
```bash
python scripts/run_trader.py --config config/trading.yaml
```

## Documentation Structure

### 1. Architecture
- System Overview
- Component Interactions
- Data Flow
- Technical Stack

### 2. API Reference
- Endpoints
- Data Models
- Authentication
- Rate Limits

### 3. Models
- Market Inefficiency Detection
- Sentiment Analysis
- Portfolio Optimization
- Risk Management

### 4. Services
- Trading Service
- Data Feed Service
- Monitoring Service
- Robo Advisory Service

### 5. Deployment
- Requirements
- Installation
- Configuration
- Monitoring
- Troubleshooting

### 6. Development
- Setup Guide
- Coding Standards
- Testing Guide
- Contributing Guide
