"""
Aggregation strategy factory for the ARGUS FL server.

Supports FedAvg (default) and FedProx.  Per-round configuration callbacks
are attached so that the server can pass hyperparameters (e.g. local_epochs)
to clients at the start of each round.
"""

import logging
from typing import Dict, Optional, Tuple

import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from server.config import ServerConfig

logger = logging.getLogger(__name__)


def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
    """
    Return per-round fit configuration sent to every client.

    Args:
        server_round: Current round number (1-indexed).

    Returns:
        Config dict; clients receive this as the `config` arg in fit().
    """
    return {
        "server_round": server_round,
        "local_epochs": 5,
    }


def on_evaluate_config_fn(server_round: int) -> Dict[str, Scalar]:
    """
    Return per-round evaluate configuration sent to every client.

    Args:
        server_round: Current round number (1-indexed).

    Returns:
        Config dict; clients receive this as the `config` arg in evaluate().
    """
    return {
        "server_round": server_round,
    }


def create_strategy(config: ServerConfig) -> fl.server.strategy.Strategy:
    """
    Instantiate the aggregation strategy specified in config.STRATEGY.

    Supported strategies:

    - ``"fedavg"`` — Federated Averaging (McMahan et al., 2017).
      Default choice; works well when clients have similar data distributions.

    - ``"fedprox"`` — Federated Proximal (Li et al., 2020).
      Adds a proximal term (mu) to the local objective to handle heterogeneous
      client data.  Use when ARGUS devices see very different scene types.

    Args:
        config: ServerConfig instance with strategy parameters.

    Returns:
        Configured Flower strategy instance.

    Raises:
        ValueError: If config.STRATEGY is not a recognised value.
    """
    common_kwargs = dict(
        fraction_fit=config.FRACTION_FIT,
        fraction_evaluate=config.FRACTION_EVALUATE,
        min_fit_clients=config.MIN_FIT_CLIENTS,
        min_evaluate_clients=config.MIN_EVALUATE_CLIENTS,
        min_available_clients=config.MIN_CLIENTS,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
    )

    strategy_name = config.STRATEGY.lower()

    if strategy_name == "fedavg":
        logger.info(
            "Using FedAvg strategy (min_clients=%d, fraction_fit=%.2f)",
            config.MIN_CLIENTS,
            config.FRACTION_FIT,
        )
        return fl.server.strategy.FedAvg(**common_kwargs)

    elif strategy_name == "fedprox":
        logger.info(
            "Using FedProx strategy (mu=%.4f, min_clients=%d)",
            config.FEDPROX_MU,
            config.MIN_CLIENTS,
        )
        return fl.server.strategy.FedProx(
            proximal_mu=config.FEDPROX_MU,
            **common_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown FL strategy: '{config.STRATEGY}'. "
            "Supported: 'fedavg', 'fedprox'"
        )
