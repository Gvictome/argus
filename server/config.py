"""
Configuration for the ARGUS Federated Learning server.

All settings are readable from environment variables so the server can be
deployed via Docker or a simple shell script without code changes.

Example environment::

    FL_PORT=8080
    NUM_ROUNDS=10
    MIN_CLIENTS=2
    STRATEGY=fedavg
"""

import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """
    FL server configuration.

    Attributes:
        FL_PORT: Port the Flower gRPC server listens on.
        NUM_ROUNDS: Total number of FL aggregation rounds.
        MIN_CLIENTS: Minimum connected clients before a round can start.
        MIN_FIT_CLIENTS: Minimum clients that must participate in fit().
        MIN_EVALUATE_CLIENTS: Minimum clients for evaluate().
        FRACTION_FIT: Fraction of available clients used for fit().
        FRACTION_EVALUATE: Fraction of available clients used for evaluate().
        STRATEGY: Aggregation strategy — "fedavg" or "fedprox".
        FEDPROX_MU: Proximal term coefficient for FedProx (ignored for FedAvg).
    """

    FL_PORT: int = 8080
    NUM_ROUNDS: int = 10
    MIN_CLIENTS: int = 2
    MIN_FIT_CLIENTS: int = 2
    MIN_EVALUATE_CLIENTS: int = 1
    FRACTION_FIT: float = 1.0
    FRACTION_EVALUATE: float = 0.5
    STRATEGY: str = "fedavg"
    FEDPROX_MU: float = 0.01

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """
        Build a ServerConfig from environment variables.

        Each field maps to an env var of the same name.  Missing vars use
        the dataclass defaults.

        Returns:
            Populated ServerConfig instance.
        """
        return cls(
            FL_PORT=int(os.getenv("FL_PORT", 8080)),
            NUM_ROUNDS=int(os.getenv("NUM_ROUNDS", 10)),
            MIN_CLIENTS=int(os.getenv("MIN_CLIENTS", 2)),
            MIN_FIT_CLIENTS=int(os.getenv("MIN_FIT_CLIENTS", 2)),
            MIN_EVALUATE_CLIENTS=int(os.getenv("MIN_EVALUATE_CLIENTS", 1)),
            FRACTION_FIT=float(os.getenv("FRACTION_FIT", 1.0)),
            FRACTION_EVALUATE=float(os.getenv("FRACTION_EVALUATE", 0.5)),
            STRATEGY=os.getenv("STRATEGY", "fedavg").lower(),
            FEDPROX_MU=float(os.getenv("FEDPROX_MU", 0.01)),
        )
