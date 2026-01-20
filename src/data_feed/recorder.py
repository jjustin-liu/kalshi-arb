"""Data recorder for persisting market data to Parquet files."""

import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import (
    KalshiOrderbook,
    KalshiTrade,
    KalshiMarket,
    UnderlyingTick,
    UnderlyingOrderbook,
    ArbitrageSignal,
    BacktestTrade,
)

logger = logging.getLogger(__name__)


class DataRecorder:
    """Records market data to Parquet files."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for batching writes
        self._kalshi_orderbook_buffer: list[dict] = []
        self._kalshi_trade_buffer: list[dict] = []
        self._underlying_tick_buffer: list[dict] = []
        self._signal_buffer: list[dict] = []

        self._buffer_size = 1000  # Flush every N records

    def record_kalshi_orderbook(self, orderbook: KalshiOrderbook):
        """Record Kalshi orderbook snapshot."""
        record = {
            "timestamp": orderbook.timestamp,
            "market_ticker": orderbook.market_ticker,
            "best_bid": orderbook.best_bid,
            "best_ask": orderbook.best_ask,
            "mid_price": orderbook.mid_price,
            "spread": orderbook.spread,
            "bid_depth_1": orderbook.yes_bids[0].quantity if orderbook.yes_bids else 0,
            "ask_depth_1": orderbook.yes_asks[0].quantity if orderbook.yes_asks else 0,
            "total_bid_depth": sum(lvl.quantity for lvl in orderbook.yes_bids),
            "total_ask_depth": sum(lvl.quantity for lvl in orderbook.yes_asks),
            "num_bid_levels": len(orderbook.yes_bids),
            "num_ask_levels": len(orderbook.yes_asks),
        }
        self._kalshi_orderbook_buffer.append(record)

        if len(self._kalshi_orderbook_buffer) >= self._buffer_size:
            self._flush_kalshi_orderbook()

    def record_kalshi_trade(self, trade: KalshiTrade):
        """Record Kalshi trade."""
        record = {
            "timestamp": trade.timestamp,
            "market_ticker": trade.market_ticker,
            "price": trade.price,
            "quantity": trade.quantity,
            "side": trade.taker_side.value,
        }
        self._kalshi_trade_buffer.append(record)

        if len(self._kalshi_trade_buffer) >= self._buffer_size:
            self._flush_kalshi_trades()

    def record_underlying_tick(self, tick: UnderlyingTick):
        """Record underlying market tick."""
        record = {
            "timestamp": tick.timestamp,
            "symbol": tick.symbol,
            "price": tick.price,
            "size": tick.size,
            "bid_price": tick.bid_price,
            "bid_size": tick.bid_size,
            "ask_price": tick.ask_price,
            "ask_size": tick.ask_size,
        }
        self._underlying_tick_buffer.append(record)

        if len(self._underlying_tick_buffer) >= self._buffer_size:
            self._flush_underlying_ticks()

    def record_signal(self, signal: ArbitrageSignal):
        """Record arbitrage signal."""
        record = {
            "timestamp": signal.timestamp,
            "market_ticker": signal.market_ticker,
            "underlying_symbol": signal.underlying_symbol,
            "underlying_price": signal.underlying_price,
            "strike_price": signal.strike_price,
            "kalshi_mid": signal.kalshi_mid,
            "fair_probability": signal.fair_probability,
            "implied_probability": signal.implied_probability,
            "basis": signal.basis,
            "expected_fees": signal.expected_fees,
            "expected_slippage": signal.expected_slippage,
            "net_edge": signal.net_edge,
            "toxicity_score": signal.toxicity_score,
            "volatility": signal.volatility,
            "time_to_expiry": signal.time_to_expiry,
            "side": signal.side.value,
            "recommended_size": signal.recommended_size,
            "confidence": signal.confidence,
            "is_tradeable": signal.is_tradeable,
        }
        self._signal_buffer.append(record)

        if len(self._signal_buffer) >= self._buffer_size:
            self._flush_signals()

    def flush_all(self):
        """Flush all buffers to disk."""
        self._flush_kalshi_orderbook()
        self._flush_kalshi_trades()
        self._flush_underlying_ticks()
        self._flush_signals()

    def _flush_kalshi_orderbook(self):
        """Flush Kalshi orderbook buffer to Parquet."""
        if not self._kalshi_orderbook_buffer:
            return

        df = pd.DataFrame(self._kalshi_orderbook_buffer)
        self._append_to_parquet(df, "kalshi_orderbook")
        self._kalshi_orderbook_buffer.clear()

    def _flush_kalshi_trades(self):
        """Flush Kalshi trades buffer to Parquet."""
        if not self._kalshi_trade_buffer:
            return

        df = pd.DataFrame(self._kalshi_trade_buffer)
        self._append_to_parquet(df, "kalshi_trades")
        self._kalshi_trade_buffer.clear()

    def _flush_underlying_ticks(self):
        """Flush underlying ticks buffer to Parquet."""
        if not self._underlying_tick_buffer:
            return

        df = pd.DataFrame(self._underlying_tick_buffer)
        self._append_to_parquet(df, "underlying_ticks")
        self._underlying_tick_buffer.clear()

    def _flush_signals(self):
        """Flush signals buffer to Parquet."""
        if not self._signal_buffer:
            return

        df = pd.DataFrame(self._signal_buffer)
        self._append_to_parquet(df, "signals")
        self._signal_buffer.clear()

    def _append_to_parquet(self, df: pd.DataFrame, name: str):
        """Append DataFrame to date-partitioned Parquet file."""
        if df.empty:
            return

        # Partition by date
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        for dt, group in df.groupby("date"):
            date_str = dt.strftime("%Y-%m-%d")
            file_path = self.data_dir / f"{name}_{date_str}.parquet"

            # Drop the partition column before writing
            group = group.drop(columns=["date"])
            table = pa.Table.from_pandas(group)

            if file_path.exists():
                # Append to existing file
                existing = pq.read_table(file_path)
                combined = pa.concat_tables([existing, table])
                pq.write_table(combined, file_path, compression="snappy")
            else:
                pq.write_table(table, file_path, compression="snappy")

            logger.debug(f"Wrote {len(group)} records to {file_path}")


class DataLoader:
    """Loads historical data from Parquet files."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_kalshi_orderbook(
        self,
        start: date,
        end: date,
        market_ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load Kalshi orderbook data."""
        df = self._load_date_range("kalshi_orderbook", start, end)
        if market_ticker and not df.empty:
            df = df[df["market_ticker"] == market_ticker]
        return df

    def load_kalshi_trades(
        self,
        start: date,
        end: date,
        market_ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load Kalshi trade data."""
        df = self._load_date_range("kalshi_trades", start, end)
        if market_ticker and not df.empty:
            df = df[df["market_ticker"] == market_ticker]
        return df

    def load_underlying_ticks(
        self,
        start: date,
        end: date,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load underlying tick data."""
        df = self._load_date_range("underlying_ticks", start, end)
        if symbol and not df.empty:
            df = df[df["symbol"] == symbol]
        return df

    def load_signals(
        self,
        start: date,
        end: date,
        market_ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load signal data."""
        df = self._load_date_range("signals", start, end)
        if market_ticker and not df.empty:
            df = df[df["market_ticker"] == market_ticker]
        return df

    def _load_date_range(self, name: str, start: date, end: date) -> pd.DataFrame:
        """Load data for a date range."""
        dfs = []
        current = start

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = self.data_dir / f"{name}_{date_str}.parquet"

            if file_path.exists():
                df = pd.read_parquet(file_path)
                dfs.append(df)

            current = date(
                current.year,
                current.month,
                current.day + 1 if current.day < 28 else 1,
            )
            # Properly increment date
            from datetime import timedelta
            current = start + timedelta(days=(current - start).days + 1)
            if current > end:
                break
            current = start
            delta = (end - start).days + 1
            break

        # Re-implement date iteration properly
        dfs = []
        from datetime import timedelta
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = self.data_dir / f"{name}_{date_str}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                dfs.append(df)
            current = current + timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined

    def get_available_dates(self, name: str) -> list[date]:
        """Get list of dates with available data."""
        dates = []
        for f in self.data_dir.glob(f"{name}_*.parquet"):
            try:
                date_str = f.stem.replace(f"{name}_", "")
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                dates.append(dt)
            except ValueError:
                continue
        return sorted(dates)


class BacktestRecorder:
    """Records backtest results."""

    def __init__(self, output_dir: str = "data/backtest"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._trades: list[dict] = []

    def record_trade(self, trade: BacktestTrade):
        """Record a backtest trade."""
        self._trades.append({
            "timestamp": trade.timestamp,
            "market_ticker": trade.market_ticker,
            "side": trade.side.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "signal_edge": trade.signal_edge,
            "realized_edge": trade.realized_edge,
            "slippage": trade.slippage,
            "fees": trade.fees,
            "pnl": trade.pnl,
            "holding_period": trade.holding_period,
            "is_winner": trade.is_winner,
        })

    def save(self, run_id: str):
        """Save backtest results."""
        if not self._trades:
            logger.warning("No trades to save")
            return

        df = pd.DataFrame(self._trades)
        file_path = self.output_dir / f"backtest_{run_id}.parquet"
        df.to_parquet(file_path, compression="snappy")
        logger.info(f"Saved {len(self._trades)} trades to {file_path}")

    def load(self, run_id: str) -> pd.DataFrame:
        """Load backtest results."""
        file_path = self.output_dir / f"backtest_{run_id}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Backtest results not found: {file_path}")
        return pd.read_parquet(file_path)
