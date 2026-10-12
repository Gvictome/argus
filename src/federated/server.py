"""
Flower Server for ARGUS Federated Learning.

This script starts the central Flower server (SuperLink) that coordinates
multiple ARGUS nodes.
"""

import flwr as fl

from src.federated.robust_strategy import RobustFedAvg

def start_server(port: int = 8080, num_rounds: int = 3, aggregation: str = "robust"):
    """Start the Flower server, screening updates unless aggregation='fedavg'."""

    strategy = RobustFedAvg(
        aggregation=aggregation,
        fraction_fit=1.0,  # Sample all available nodes for training
        fraction_evaluate=0.5,  # Sample 50% for evaluation
        min_fit_clients=1,
        min_available_clients=1,
    )

    print(f"[*] ARGUS FL Server starting on port {port}...")
    print(f"[*] Training for {num_rounds} rounds, aggregation={aggregation}.")

    # Start the server
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
