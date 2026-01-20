"""
Risk Management Module.

Provides toxic flow detection and position management to avoid
adverse selection when market making.

Key Components:
- OFICalculator: Order Flow Imbalance from orderbook changes
- VPINCalculator: Volume-Synchronized Probability of Informed Trading
- SweepDetector: Detects aggressive level clearing
- ToxicityMonitor: Combines all signals into single toxicity score
- PositionTracker: Tracks positions and P&L across markets
- RiskManager: Main coordinator for risk checks

Usage:
    from src.risk import RiskManager, RiskConfig

    risk_mgr = RiskManager(RiskConfig(), initial_capital=10000)

    # Feed market data
    risk_mgr.update_orderbook(orderbook)
    risk_mgr.update_trade(trade)

    # Evaluate signal
    assessment = risk_mgr.evaluate_signal(signal, orderbook)
    if assessment.approved:
        # Execute with assessment.adjusted_size
        pass
"""

from .config import RiskConfig, RiskLimits, ToxicityConfig
from .ofi import OFICalculator, OFIState
from .vpin import VPINCalculator, VPINState
from .sweep_detector import SweepDetector, SweepEvent, SweepState
from .toxicity_monitor import ToxicityMonitor
from .position_limits import PositionTracker, PositionState
from .risk_manager import RiskManager, RiskAssessment, RejectionReason

__all__ = [
    # Config
    "RiskConfig",
    "RiskLimits",
    "ToxicityConfig",
    # OFI
    "OFICalculator",
    "OFIState",
    # VPIN
    "VPINCalculator",
    "VPINState",
    # Sweep
    "SweepDetector",
    "SweepEvent",
    "SweepState",
    # Toxicity
    "ToxicityMonitor",
    # Position
    "PositionTracker",
    "PositionState",
    # Risk Manager
    "RiskManager",
    "RiskAssessment",
    "RejectionReason",
]
