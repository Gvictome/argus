"""
FederatedAgent - Manages Federated Learning for ARGUS
"""

import logging
import time
from src.agents import BaseAgent, AgentType
from src.federated.client import ArgusFlowerClient, start_client
from src.detection.models import ModelManager
from src.training import LocalTrainer
from src.config import Settings

logger = logging.getLogger(__name__)

class FederatedAgent(BaseAgent):
    """
    Agent responsible for orchestrating Flower Federated Learning rounds.
    """

    def __init__(self):
        super().__init__(name="federated", type=AgentType.DETECTION)
        self.settings = Settings()
        self.model_manager = ModelManager()
        self.trainer = None
        self.client = None

    def start(self):
        """Start the federated learning process"""
        print("[FederatedAgent] Initializing local model and trainer...")
        
        # Load model
        self.model_manager.load_model()
        
        # Initialize trainer
        self.trainer = LocalTrainer(self.model_manager, self.settings)
        
        # Create Flower client
        self.client = ArgusFlowerClient(self.model_manager, self.trainer, self.settings)
        
        print(f"[FederatedAgent] Connecting to FL server at {self.settings.FL_SERVER_URL}...")
        try:
            start_client(self.settings.FL_SERVER_URL, self.client)
        except Exception as e:
            print(f"[FederatedAgent] Error during FL session: {e}")
            logger.error("FL error", exc_info=True)

    def run(self):
        """Keep agent alive and start training"""
        self.start()
        while True:
            time.sleep(1)
