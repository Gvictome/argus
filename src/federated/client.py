"""
Flower (flwr) NumPy client for ARGUS Federated Learning.

ArgusFlowerClient implements the standard Flower NumPyClient interface so that
each ARGUS node can participate in a federated round orchestrated by the
central Flower server (server/app.py).

Usage::

    from src.detection.models import ModelManager
    from src.training import LocalTrainer
    from src.federated.client import ArgusFlowerClient, start_client
    from src.config import settings

    model_manager = ModelManager()
    model_manager.load_model()
    trainer = LocalTrainer(model_manager, settings)
    client = ArgusFlowerClient(model_manager, trainer, settings)
    start_client(settings.FL_SERVER_URL, client)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    _FLWR_AVAILABLE = True
except ImportError:
    fl = None  # type: ignore[assignment]
    _FLWR_AVAILABLE = False
    logger.warning(
        "flwr not installed — ArgusFlowerClient will raise on use. "
        "Install with: pip install flwr>=1.13.0"
    )

# Base class: fl.client.NumPyClient when flwr is available; object otherwise
_BaseClient = fl.client.NumPyClient if _FLWR_AVAILABLE else object


class ArgusFlowerClient(_BaseClient):
    """
    Flower NumPy client that wraps ARGUS model management and local training.

    Participates in FL rounds by:
    1. Receiving global weights from the server (fit / evaluate).
    2. Injecting them into the local YOLOv8n model.
    3. Running a local fine-tuning epoch (fit only).
    4. Returning updated weights + sample count + metrics to the server.
    """

    def __init__(self, model_manager, trainer, config) -> None:
        """
        Args:
            model_manager: ModelManager instance (src.detection.models).
            trainer: LocalTrainer instance (src.training).
            config: Settings instance (src.config.Settings).
        """
        self.model_manager = model_manager
        self.trainer = trainer
        self.config = config

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return the current local model weights to the server.

        Called by Flower before fit() in some strategies.

        Args:
            config: Round configuration dict from the server (may be empty).

        Returns:
            Ordered list of numpy weight arrays.
        """
        logger.debug("get_parameters called")
        return self.model_manager.get_model_weights(self.model_manager.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global weights, run local training, return updated weights.

        Args:
            parameters: Global model weights from the server.
            config: Per-round configuration dict (may contain "epochs", etc.).

        Returns:
            Tuple of (updated_weights, num_samples, metrics_dict).
        """
        logger.info("FL fit() — injecting global weights and running local training")

        # Inject global weights
        self.model_manager.set_model_weights(self.model_manager.model, parameters)

        # Run local fine-tuning
        epochs = int(config.get("local_epochs", self.config.FL_LOCAL_EPOCHS))
        metrics = self.trainer.train_epoch(
            self.model_manager.model,
            epochs=epochs,
        )

        # Return updated local weights
        updated_weights = self.model_manager.get_model_weights(self.model_manager.model)
        num_samples = len(self.trainer.get_training_data())

        logger.info(
            "FL fit() complete — samples=%d metrics=%s", num_samples, metrics
        )
        return updated_weights, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Receive global weights and evaluate them on the local validation set.

        Args:
            parameters: Global model weights from the server.
            config: Per-round evaluation config dict.

        Returns:
            Tuple of (loss, num_samples, metrics_dict).
        """
        logger.info("FL evaluate() — injecting global weights and evaluating")

        self.model_manager.set_model_weights(self.model_manager.model, parameters)

        loss, accuracy = self.trainer.evaluate(self.model_manager.model)
        num_samples = len(self.trainer.get_training_data())

        logger.info(
            "FL evaluate() complete — loss=%.4f accuracy=%.4f samples=%d",
            loss,
            accuracy,
            num_samples,
        )
        return float(loss), num_samples, {"accuracy": float(accuracy)}


def start_client(server_url: str, client: ArgusFlowerClient) -> None:
    """
    Connect to the Flower server and start the federated learning loop.

    This call blocks until the server closes the connection (i.e. all FL
    rounds are complete).

    Args:
        server_url: gRPC address of the Flower server, e.g. "192.168.1.10:8080".
        client: Configured ArgusFlowerClient instance.

    Raises:
        RuntimeError: If flwr is not installed.
    """
    if not _FLWR_AVAILABLE:
        raise RuntimeError(
            "flwr is required to start the FL client. "
            "Install with: pip install flwr>=1.13.0"
        )

    logger.info("Connecting to FL server at %s", server_url)
    fl.client.start_numpy_client(
        server_address=server_url,
        client=client,
    )
    logger.info("FL client finished")
