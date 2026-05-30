"""
Orchestrator patch to extract and log reward ledger data.

This module patches the orchestrator's monitor to extract ledger data
from environment states and log it to W&B and CSV.
"""

from typing import Any, Dict, List

# Try to import ledger logging utilities
try:
    from dakota_rl_training.utils.ledger_logging import extract_ledger_from_info
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False
    print("Warning: Ledger logging utilities not available")


def extract_ledger_from_states(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract ledger data from verifiers state dicts.
    
    The verifiers framework passes state dicts that may contain ledger data
    either directly or nested in 'info' or 'ledger' keys.
    """
    ledger_infos = []
    for state in states:
        if not isinstance(state, dict):
            continue
        
        # Check multiple possible locations for ledger data
        ledger = None
        
        # Direct ledger key
        if "ledger" in state and isinstance(state["ledger"], dict):
            ledger = state["ledger"]
        
        # In info dict
        elif "info" in state and isinstance(state["info"], dict):
            ledger = extract_ledger_from_info(state["info"])
        
        # State itself might be the ledger (if passed directly)
        elif any(k.startswith(("char_", "morph_", "w_", "composite_", "reward_scalar")) for k in state.keys()):
            ledger = extract_ledger_from_info(state)
        
        if ledger:
            ledger_infos.append(ledger)
    
    return ledger_infos


def patch_orchestrator_monitor_logging():
    """
    Patch the orchestrator's monitor to extract and log ledger data.
    
    This should be called before starting the orchestrator.
    """
    if not LEDGER_AVAILABLE:
        return
    
    # We'll hook into the orchestrator's step logging
    # The actual integration happens in the orchestrator code modification
    pass

