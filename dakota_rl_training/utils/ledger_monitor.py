"""
Custom Monitor for Reward Ledger Logging

Extends WandbMonitor to also log detailed reward ledger data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from prime_rl.utils.monitor import WandbMonitor
from prime_rl.utils.config import WandbMonitorConfig
from prime_rl.utils.logger import get_logger

# Import ledger logging utilities
try:
    from dakota_rl_training.utils.ledger_logging import log_step_ledger, extract_ledger_from_info
    LEDGER_LOGGING_AVAILABLE = True
except ImportError:
    LEDGER_LOGGING_AVAILABLE = False
    get_logger().warning("Ledger logging utilities not available. Install dakota_rl_training package.")


class LedgerWandbMonitor(WandbMonitor):
    """
    Extended WandbMonitor that also logs reward ledger data.
    
    This monitor extracts ledger data from environment states and logs
    detailed reward component breakdowns to W&B and CSV.
    """
    
    def __init__(
        self,
        config: WandbMonitorConfig | None,
        output_dir: Path | None = None,
        tokenizer=None,
        run_config=None,
    ):
        super().__init__(config, output_dir, tokenizer, run_config)
        self.logger = get_logger()
        self.ledger_enabled = LEDGER_LOGGING_AVAILABLE and self.enabled and self.is_master
        
        if self.ledger_enabled:
            self.logger.info("Ledger logging enabled - will log detailed reward components")
        elif not LEDGER_LOGGING_AVAILABLE:
            self.logger.warning("Ledger logging utilities not available - skipping ledger logging")
    
    def log_with_ledger(
        self,
        metrics: Dict[str, Any],
        states: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Log metrics and extract ledger data from states if available.
        
        Args:
            metrics: Standard metrics dict to log
            states: Optional list of state dicts from environment steps (may contain ledger data)
        """
        # Log standard metrics
        self.log(metrics)
        
        # Extract and log ledger data if available
        if self.ledger_enabled and states:
            step = metrics.get("step")
            if step is not None:
                # Extract ledger data from states
                ledger_infos = []
                for state in states:
                    # Check if ledger is in state dict
                    if isinstance(state, dict):
                        ledger = extract_ledger_from_info(state)
                        if ledger:
                            ledger_infos.append(ledger)
                
                # Log ledger if we found any
                if ledger_infos:
                    log_step_ledger(step, ledger_infos, wandb_log=True)


def setup_monitor(
    config: WandbMonitorConfig | None,
    output_dir: Path | None = None,
    tokenizer=None,
    run_config=None,
) -> LedgerWandbMonitor:
    """
    Setup monitor with ledger logging support.
    
    This replaces the standard setup_monitor to use LedgerWandbMonitor.
    """
    monitor = LedgerWandbMonitor(config=config, output_dir=output_dir, tokenizer=tokenizer, run_config=run_config)
    return monitor

