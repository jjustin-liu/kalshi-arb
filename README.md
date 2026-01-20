# Kalshi Cross-Asset Basis Arbitrage Engine

A real-time trading system that exploits pricing discrepancies between Kalshi prediction markets (binary options on S&P 500 levels) and ES futures.

## Overview

The system identifies arbitrage opportunities by comparing:
- **Fair probability**: Black-Scholes valuation using ES futures price, strike, volatility, and time to expiry
- **Implied probability**: Kalshi orderbook mid-price / 100

When `fair_prob - implied_prob > threshold`, the system generates a trade signal.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Kalshi WS      │     │  Databento      │
│  (Orderbooks)   │     │  (ES Futures)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│            Data Feed Layer              │
│  KalshiOrderbook, UnderlyingTick        │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│     Pricing     │     │      Risk       │
│  - BinaryPricer │     │  - OFI          │
│  - VolSurface   │     │  - VPIN         │
│  - BasisCalc    │     │  - Toxicity     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │   Execution Engine  │
         │  - Liquidity Check  │
         │  - TWAP/Iceberg     │
         │  - Order Manager    │
         └─────────────────────┘
```

## Features

### Risk Management
- **Order Flow Imbalance (OFI)**: Detects buying/selling pressure from orderbook changes
- **VPIN**: Volume-synchronized probability of informed trading
- **Sweep Detection**: Identifies aggressive level clearing
- **Position Limits**: Per-market, total, and correlated position controls

### Execution
- **Liquidity Analysis**: Spread, depth, and market impact checks
- **Smart Order Routing**: TWAP and Iceberg execution for large orders
- **Simulated Delta Hedging**: Tracks theoretical ES hedge position

### Backtesting
- **Event-Driven Simulation**: Same code path as live trading
- **Multiple Fill Models**: Simple, queue-aware, and market impact
- **Walk-Forward Optimization**: Parameter tuning with out-of-sample validation

### Monitoring
- **Real-Time Dashboard**: FastAPI + WebSocket
- **Prometheus Metrics**: Signals, trades, P&L, toxicity
- **Alerting**: Toxicity, drawdown, and spread alerts

## Installation

```bash
# Clone the repository
git clone https://github.com/jjustin-liu/kalshi-arb.git
cd kalshi-arb

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API credentials
```

## Configuration

### Required API Keys

1. **Kalshi**: Sign up at [kalshi.com](https://kalshi.com) and generate API keys
2. **Databento**: Sign up at [databento.com](https://databento.com) for ES futures data

Add credentials to `.env`:
```bash
KALSHI_API_KEY=your_key
KALSHI_API_SECRET=your_secret
DATABENTO_API_KEY=your_key
```

## Usage

### Collect Data (Run First)
```bash
# Collect market data for backtesting (run for 1-2 weeks)
python -m src.cli collect --markets INXD --output data
```

### Backtest
```bash
# Run backtest on historical data
python -m src.cli backtest --start 2024-01-01 --end 2024-01-31 --output report.html
```

### Optimize Parameters
```bash
# Grid search optimization
python -m src.cli optimize --method grid --metric sharpe

# Walk-forward optimization
python -m src.cli optimize --method walk-forward --metric sharpe
```

### Live Trading
```bash
# Paper trading mode
python -m src.cli live --paper --markets INXD

# Live trading (requires funded Kalshi account)
python -m src.cli live --markets INXD
```

### Monitoring Dashboard
```bash
# Start web dashboard
python -m src.cli monitor --port 8080
# Open http://localhost:8080
```

## Project Structure

```
kalshi-arb/
├── src/
│   ├── data_feed/          # Kalshi/Databento clients, schemas
│   ├── pricing/            # Black-Scholes, volatility, basis calculator
│   ├── risk/               # OFI, VPIN, toxicity, position limits
│   ├── execution/          # Liquidity, orders, TWAP/Iceberg
│   ├── backtest/           # Event engine, fill models, optimizer
│   ├── monitoring/         # Dashboard, alerts, metrics
│   └── cli.py              # Command-line interface
├── scripts/
│   ├── collect_kalshi.py   # Kalshi data recorder
│   ├── collect_databento.py # ES futures recorder
│   └── run_collector.py    # Combined collector
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
└── data/                   # Recorded market data (gitignored)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Algorithms

### VPIN (Volume-Synchronized PIN)
Detects informed trading by measuring order flow imbalance in volume buckets:
```
VPIN = |Σ(buy_volume) - Σ(sell_volume)| / Σ(total_volume)
```
High VPIN (>0.7) indicates potential adverse selection.

### Order Flow Imbalance
Measures buying/selling pressure from orderbook changes:
```
OFI = Δ(bid_depth) - Δ(ask_depth)
```
Positive OFI = buying pressure, negative = selling pressure.

### Toxicity Score
Combined signal for trade timing:
```
toxicity = 0.25×OFI + 0.30×VPIN + 0.20×spread + 0.15×sweep + 0.10×imbalance
```
Pause trading when toxicity > 0.6.

## License

MIT

## Disclaimer

This software is for educational purposes only. Trading prediction markets and futures involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
