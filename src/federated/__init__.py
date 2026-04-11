"""
Federated Learning package for ARGUS.

Provides the Flower client (ArgusFlowerClient), FL scheduler, and shared
weight compression utilities used by the client during FL rounds.
"""

from src.federated.client import ArgusFlowerClient, start_client
from src.federated.scheduler import FLScheduler

__all__ = ["ArgusFlowerClient", "start_client", "FLScheduler"]
