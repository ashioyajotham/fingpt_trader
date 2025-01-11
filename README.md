# FinGPT Market Inefficiency Trading System

A sophisticated crypto trading system combining FinGPT with market inefficiency detection and robo-advisory capabilities.

## Market Inefficiencies in Crypto Markets

Market inefficiencies represent deviations from the Efficient Market Hypothesis (EMH) where prices don't fully reflect all available information. In crypto markets, these inefficiencies are particularly pronounced due to:

### Information Asymmetry
```math
IA(t) = |P(t) - E[P(t)|Ω(t)]|
where:
- P(t): Observed price at time t
- Ω(t): Complete information set
- E[P(t)|Ω(t)]: Expected price given full information
```

Crypto markets often exhibit significant information gaps due to:
- Fragmented exchange landscapes
- Varying regulatory environments
- Complex technical foundations
- Rapid technological evolution

### Market Microstructure Inefficiencies
Common in crypto due to:
1. **Order Book Fragmentation**
   - Multiple exchanges
   - Varying liquidity pools
   - Cross-chain complexities

2. **Price Formation Dynamics**
   ```math
   P(t) = P(t-1) + λ(D(t) + η(t))
   where:
   - λ: Market impact parameter
   - D(t): Order flow imbalance
   - η(t): Noise trader impact
   ```

3. **Arbitrage Opportunities**
   - Cross-exchange pricing discrepancies
   - Cross-chain value transfers
   - Market maker inventory imbalances

### Why These Matter
1. **Alpha Generation**
   - Short-term price prediction
   - Statistical arbitrage opportunities
   - Mean reversion strategies

2. **Risk Management**
   - Liquidity risk assessment
   - Market impact estimation
   - Portfolio rebalancing costs

3. **Market Making**
   - Optimal spread determination
   - Inventory management
   - Cross-exchange arbitrage

## Core Components

### 1. Market Inefficiency Detection
We focus on four key types of market inefficiencies:

#### a) Sentiment-Price Divergence
When market sentiment and price movements show significant disparity:
```math
SPD(t) = |S(t) - P'(t)| > θ
where:
- S(t): Normalized sentiment score at time t
- P'(t): Normalized price change
- θ: Divergence threshold
```

#### b) Order Flow Imbalance
Detects unusual buying/selling pressure:
```math
OFI(t) = Σ[V_b(i) - V_a(i)] / Σ[V_b(i) + V_a(i)]
where:
- V_b(i): Volume at bid level i
- V_a(i): Volume at ask level i
```

#### c) Microstructure Mean Reversion
Based on mean-reverting behavior in market microstructure:
```math
MR(t) = -λ(P(t) - μ) + ε(t)
where:
- λ: Mean reversion rate
- μ: Long-term mean price
- ε(t): Random noise
```

#### d) Cross-Exchange Arbitrage
Identifies price discrepancies across exchanges:
```math
XA(t) = max(|P_i(t) - P_j(t)| - c_ij)
where:
- P_i(t): Price on exchange i
- P_j(t): Price on exchange j
- c_ij: Transaction costs
```

### 2. Robo Advisory System
Our robo advisor implements modern portfolio theory with ESG and tax considerations:

#### a) Portfolio Optimization
```math
w* = argmax_w(w'μ - λw'Σw)
subject to:
- Σw_i = 1
- w_i ≥ 0
- ESG_score(w) ≥ threshold
```

#### b) Tax-Loss Harvesting
```math
TaxSavings = Losses × TaxRate
Execute if: TaxSavings > TransactionCosts + OpportunityCost
```

#### c) Dynamic Rebalancing
```math
Rebalance if: |w_current - w_target| > min(base_threshold, tax_adjusted_threshold)
where: tax_adjusted_threshold = base_threshold × (1 + tax_impact_factor)
```

## System Architecture

```mermaid
graph TB
    subgraph Data Layer
        MD[Market Data Stream] --> PP[Price Processor]
        NF[News Feed] --> NP[News Processor]
        OB[Order Book] --> OP[Order Flow Processor]
        XE[Cross-Exchange Data] --> XA[Arbitrage Detector]
    end

    subgraph Analysis Layer
        PP --> TA[Technical Analysis]
        NP --> SA[Sentiment Analysis]
        OP --> MA[Microstructure Analysis]
        XA --> CA[Cross-Exchange Analysis]
        
        subgraph LLM Module
            SA --> FG[FinGPT]
            FG --> SS[Sentiment Signals]
        end
        
        subgraph Inefficiency Detection
            TA --> ID[Inefficiency Detector]
            MA --> ID
            SS --> ID
            CA --> ID
        end

        subgraph Robo Advisory
            CP[Client Profiles] --> RA[Risk Assessment]
            RA --> PA[Portfolio Analysis]
            PA --> PO[Portfolio Optimization]
            PA --> TH[Tax Harvesting]
            PA --> ESG[ESG Optimization]
            PO --> RB[Rebalancing]
            TH --> RB
            ESG --> RB
        end
    end

    subgraph Execution Layer
        ID --> SG[Signal Generator]
        RB --> SG
        SG --> PS[Position Sizing]
        PS --> RM[Risk Management]
        RM --> OE[Order Execution]
    end

    subgraph Monitoring
        OE --> PT[Performance Tracker]
        PT --> RA[Risk Analyzer]
        RA --> AA[Alpha Attribution]
        PT --> TC[Tax Calculator]
        PT --> EC[ESG Compliance]
    end
```

## Project Structure
```
fingpt_trader/
├── main.py                 # Main trading system
├── models/
│   ├── llm/               # LLM-related models
│   │   ├── base.py        # Base LLM interface
│   │   ├── fingpt.py      # FinGPT implementation
│   │   └── utils/
│   │       ├── tokenizer.py
│   │       └── inference.py
│   ├── market/            # Market analysis
│   │   ├── inefficiency.py
│   │   └── patterns.py
│   ├── portfolio/         # Portfolio management
│   │   ├── optimization.py
│   │   ├── risk.py
│   │   └── rebalancing.py
│   └── sentiment/
│       ├── analyzer.py    # Sentiment analysis
│       └── preprocessor.py
├── services/
│   ├── base_service.py    # Base service interface
│   ├── data_feeds/        # Market data services
│   │   ├── market_data_service.py
│   │   └── news_service.py
│   ├── monitoring/        # System monitoring
│   │   ├── system_monitor.py
│   │   └── performance_tracker.py
│   └── trading/          # Trading services
│       ├── robo_service.py
│       └── order_manager.py
├── data/                  # Data storage
│   ├── raw/              # Raw market data
│   ├── processed/        # Processed features
│   └── logs/             # System logs
├── config/               # Configuration files
│   ├── trading.yaml     # Main trading config
│   ├── test_trading.yaml # Test config
│   └── logging.yaml     # Logging config
├── scripts/             # Utility scripts
│   ├── backtest.py
│   ├── analyze.py
│   ├── live_trade.py
│   └── run_trader.py
├── examples/            # Example implementations
│   ├── market_data.py
│   ├── sentiment_analysis.py
│   ├── pair_trading.py
│   └── robo_advisor_demo.py
├── utils/              # Utility functions
│   ├── config.py      # Configuration management
│   ├── logging.py     # Logging utilities
│   └── validation.py  # Data validation
└── tests/             # Test suite
    ├── test_system_integration.py
    ├── test_portfolio_optimization.py
    ├── test_signal_generation.py
    └── test_robo_advisor.py
```

## Mathematical Foundations

### 1. Market Inefficiency Detection
```math
D(t) = [P(t) - P(t-1)]/P(t-1) - λS(t)
OFI(t) = Σ[V_b(i) - V_a(i)] / Σ[V_b(i) + V_a(i)]
```

### 2. Portfolio Optimization
```math
E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}[(τΣ)^{-1}π + P'Ω^{-1}Q]
```

### 3. Risk Management
```math
VaR_α = -inf{x ∈ ℝ: P(X ≤ x) > α}
Kelly = (p × ln(1 + b) - (1-p) × ln(1 - a))/((a + b) × ln(1 + b) × ln(1 - a))
```

## Implementation Examples

### 1. System Initialization
```python
system = TradingSystem(config_path="config/trading.yaml")
await system.initialize()
```

### 2. Client Portfolio Management
```python
# Register client
await system.register_client("client1", {
    "risk_score": 7.0,
    "investment_horizon": 10,
    "constraints": {"max_stock": 0.8},
    "tax_rate": 0.25,
    "esg_preferences": {"min_score": 0.7}
})

# Get recommendation
portfolio = await system.get_portfolio_recommendation("client1")
```

### 3. Trading Execution
```python
# Market data analysis
market_data = await system.get_market_data()
signals = await system.detect_inefficiencies(market_data)

# Generate and execute trades
trades = system.generate_trades(signals)
await system.execute_trades(trades)
```

## Configuration

### Sample Trading Configuration
```yaml
trading:
  max_position_size: 0.2
  risk_limit: 0.05
  confidence_threshold: 0.65

robo:
  rebalance_threshold: 0.05
  tax_impact_factor: 0.3
  base_threshold: 0.02
```

## Installation

```bash
# Clone repository
git clone https://github.com/ashioyajotham/fingpt_trader.git
cd fingpt_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start trading
python scripts/run_trader.py --config config/trading.yaml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.