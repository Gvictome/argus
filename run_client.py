"""
ARGUS Federated Learning Client Runner
"""

import sys
import os
import logging

# Add current directory to path so 'src' is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.detection.models import ModelManager
    from src.training import LocalTrainer
    from src.federated.client import ArgusFlowerClient, start_client
    from src.config import settings
except ImportError as e:
    print(f"Error: {e}")
    print("Could not import ARGUS modules. Make sure you are in the root 'argus' directory.")
    sys.exit(1)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("argus_client")

    logger.info("Initializing ARGUS Federated Learning Node...")
    
    # Setup model
    model_manager = ModelManager()
    model_manager.load_model()
    
    # Setup trainer
    trainer = LocalTrainer(model_manager, settings)
    
    # Setup Flower client
    client = ArgusFlowerClient(model_manager, trainer, settings)
    
    # Connect to server
    logger.info(f"Connecting to Flower server at {settings.FL_SERVER_URL}...")
    try:
        start_client(settings.FL_SERVER_URL, client)
    except Exception as e:
        logger.error(f"FL Session failed: {e}")

if __name__ == "__main__":
    main()
