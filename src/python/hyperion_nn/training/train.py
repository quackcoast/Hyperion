# this will be used for the following:
# - this is the main training script that will be used to train the neural network
# - load config, initialize the Dataset and DataLoader, and initialize the model
# - implement the training loop with forward and backward passes, loss calculation, and optimizer step
# - handle model saving and loading

import logging
import sys
import os
import hyperion_nn.config as config

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(name)-35s] [%(levelname)-8s] %(message)s",
                    handlers=[logging.FileHandler(os.path.join(config.PathsConfig.LOGS_DIR, "training.log")), logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)
