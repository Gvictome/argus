"""
ARGUS Federated Learning server entry point.

Run on a central host to orchestrate FL rounds across all ARGUS Pi nodes.

Usage::

    # Default config (reads env vars)
    python server/app.py

    # Custom config via env vars
    FL_PORT=8080 NUM_ROUNDS=20 MIN_CLIENTS=3 STRATEGY=fedprox python server/app.py

The server blocks until all NUM_ROUNDS are complete, then exits.
Clients (src/federated/client.py) connect when their scheduler fires.
"""

import logging
import sys

import flwr as fl

from server.config import ServerConfig
from server.strategy import create_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Start the Flower FL server.

    Reads configuration from environment variables, builds the aggregation
    strategy, and starts the Flower gRPC server.  Blocks until all rounds
    have completed.
    """
    config = ServerConfig.from_env()

    logger.info(
        "Starting ARGUS FL server — port=%d rounds=%d strategy=%s min_clients=%d",
        config.FL_PORT,
        config.NUM_ROUNDS,
        config.STRATEGY,
        config.MIN_CLIENTS,
    )

    strategy = create_strategy(config)

    fl.server.start_server(
        server_address=f"0.0.0.0:{config.FL_PORT}",
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
    )

    logger.info("FL server finished all %d rounds", config.NUM_ROUNDS)


if __name__ == "__main__":
    main()
