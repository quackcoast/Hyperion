# this will be used for the following:
# - this is the main training script that will be used to train the neural network
# - load config, initialize the Dataset and DataLoader, and initialize the model
# - implement the training loop with forward and backward passes, loss calculation, and optimizer step
# - handle model saving and loading

import logging
import sys
import os
import torch
import torch.optim as optim
import glob
from tqdm import tqdm

import hyperion_nn.config as config
from hyperion_nn.data_utils.dataset import ChessDataset
from hyperion_nn.models.resnet_cnn import HyperionNN
from torch.utils.data import DataLoader


# 1. Get the root logger. This is the master logger that all others inherit from.

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 2. Create the file handler. This handler writes messages to your log file.
log_file_path = os.path.join(config.PathsConfig.LOGS_DIR, "training.log")
file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
file_handler.setLevel(logging.INFO) # Log everything of level INFO and above to the file.
file_formatter = logging.Formatter("%(asctime)s [%(name)-30s] [%(levelname)-8s] %(message)s")
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# 3. Create the console handler. This handler prints messages to the console
console_handler = logging.StreamHandler(sys.stdout)
# Set a higher level for the console. WARNING means it will only print messages
# that are warnings, errors, or critical. INFO and DEBUG messages will be ignored.
console_handler.setLevel(logging.WARNING) 
console_formatter = logging.Formatter("[%(levelname)-8s] %(message)s")
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def train_model():
    '''Main function to train the HyperionNN model.'''

    logger.info("Starting training process...")
    device = config.HardwareBasedConfig.DEVICE
    logger.info(f"Using device: {device}")

    # 1) model initialization
    logger.info("Initializing model and optimizer...")
    model = HyperionNN().to(device)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=config.TrainingConfig.LEARNING_RATE,
                           weight_decay=config.TrainingConfig.WEIGHT_DECAY)
    
    # 2) checkpoint loading
    global_step = 0
    start_epoch = 0
    checkpoint_dir = config.PathsConfig.CHECKPOINT_DIR
    if os.path.exists(checkpoint_dir):
        checkpoints_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
        if checkpoints_list:
            latest_checkpoint_path = max(checkpoints_list, key=os.path.getctime)
            logger.info(f"Loading checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            start_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Checkpoint loaded. Resuming from step {global_step} (Epoch {start_epoch}).")
        else:
            logger.warning("No checkpoints found. Starting training from scratch.")

    # 3) load data
    training_dataset = ChessDataset()
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config.HardwareBasedConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=config.HardwareBasedConfig.NUM_WORKERS,
        pin_memory=True,
    )

    # 4) init loss functions
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # 5) main training loop
    logger.info(f"Starting training loop at step {global_step}...")
    model.train()
    
    steps_per_epoch = len(training_dataloader)
    total_epochs = (config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS // steps_per_epoch) + 1

    print(r"""
    +-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-+
    |                                                                           |@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |                                                                           |@ @ @ @ @ @ @ @ = + + + + + * @ @ @ @ @ @ @ @ @|
    | /$$   /$$                                         /$$                     |@ @ @ @ @ @ @ @ @ + + + + + + * @ @ @ + + + @ @|
    || $$  | $$                                        |__/                     |@ @ @ @ @ @ @ @ @ @ @ + + @ @ @ + + + + + + @ @|
    || $$  | $$ /$$   /$$  /$$$$$$   /$$$$$$   /$$$$$$  /$$  /$$$$$$  /$$$$$$$  |@ @ @ @ @ @ @ @ @ @ @ @ @ = + + + + + + + + @ @|
    || $$$$$$$$| $$  | $$ /$$__  $$ /$$__  $$ /$$__  $$| $$ /$$__  $$| $$__  $$ |@ @ @ @ @ @ @ @ @ @ = = = = = + + @ @ + + @ @ @|
    || $$__  $$| $$  | $$| $$  \ $$| $$$$$$$$| $$  \__/| $$| $$  \ $$| $$  \ $$ |@ @ @ @ @ @ @ @ = = = = = = = @ @ . @ + + @ @ @|
    || $$  | $$| $$  | $$| $$  | $$| $$_____/| $$      | $$| $$  | $$| $$  | $$ |@ @ @ @ @ @ = = = = = = = @ @ . : @ + + + @ @ @|
    || $$  | $$|  $$$$$$$| $$$$$$$/|  $$$$$$$| $$      | $$|  $$$$$$/| $$  | $$ |@ @ @ @ @ @ @ @ @ @ = @ @ . . . - @ + + @ * @ @|
    ||__/  |__/ \____  $$| $$____/  \_______/|__/      |__/ \______/ |__/  |__/ |@ @ @ @ @ @ @ @ @ @ @ @ . . . - @ = = + @ + + @|
    |           /$$  | $$| $$                                                   |@ @ @ @ @ @ - = @ @ @ . . . - - @ = = = @ + + @|
    |          |  $$$$$$/| $$                                                   |@ @ @ @ @ @ @ @ @ @ . . . - - @ = = = @ + + + @|
    |           \______/ |__/                                                   |@ @ @ @ @ @ @ @ @ . . . - - @ @ = = = @ + + + @|
    |   /$$                        /$$           /$$                            |@ @ @ @ @ @ @ @ . . . - - @ @ = = = = @ @ + + @|
    |  | $$                       |__/          |__/                            |@ @ @ @ @ @ @ . . . - - @ @ @ @ = = @ @ @ + + @|
    | /$$$$$$    /$$$$$$  /$$$$$$  /$$ /$$$$$$$  /$$ /$$$$$$$   /$$$$$$         |@ @ @ @ @ @ @ . - - - @ @ @ @ @ = = @ @ @ @ + @|
    ||_  $$_/   /$$__  $$|____  $$| $$| $$__  $$| $$| $$__  $$ /$$__  $$        |@ @ @ - @ @ - - - - @ @ @ = @ @ = @ @ @ @ @ @ @|
    |  | $$    | $$  \__/ /$$$$$$$| $$| $$  \ $$| $$| $$  \ $$| $$  \ $$        |@ @ @ - - - - - @ @ @ @ @ - @ @ = @ @ @ @ @ @ @|
    |  | $$ /$$| $$      /$$__  $$| $$| $$  | $$| $$| $$  | $$| $$  | $$        |@ @ @ @ - = - @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |  |  $$$$/| $$     |  $$$$$$$| $$| $$  | $$| $$| $$  | $$|  $$$$$$$        |@ @ @ * * + + @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |   \___/  |__/      \_______/|__/|__/  |__/|__/|__/  |__/ \____  $$        |@ @ * * * @ + + @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |                                                          /$$  \ $$        |@ * * * @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |                                                         |  $$$$$$/        |@ @ * @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    |                                                          \______/         |@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @|
    +-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-+
    """)

    for epoch in range(start_epoch, total_epochs):
        # Create a tqdm progress bar for the current epoch
        progress_bar = tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)
        
        for batch in progress_bar:
            input_planes, policy_target, value_target = batch
            
            input_planes = input_planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            
            optimizer.zero_grad()
            policy_logits, value_output = model(input_planes)

            # calculate losses
            policy_loss_indices = torch.argmax(policy_target, dim=1)
            loss_policy = policy_loss_fn(policy_logits, policy_loss_indices)
            value_prediction = torch.tanh(value_output)
            loss_value = value_loss_fn(value_prediction, value_target)
            total_loss = loss_policy + loss_value

            # backpropagation
            total_loss.backward()
            optimizer.step()
            global_step += 1

            # Update the progress bar with the latest loss information
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", step=global_step)

            if global_step % config.TrainingConfig.LOG_EVERY_N_STEPS == 0:
                logger.info(f"Step [{global_step}/{config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS}], Loss: {total_loss.item():.4f}")

            if global_step % config.TrainingConfig.VALIDATE_EVERY_N_STEPS == 0:
                # TODO: Implement the validation logic here
                model.train()

            if global_step % config.TrainingConfig.SAVE_CHECKPOINTS_EVERY_N_STEPS == 0:
                checkpoint_name = f"checkpoint_step_{global_step}.pt"
                # Make sure the checkpoint path exists and is correct
                checkpoint_path = os.path.join(config.PathsConfig.CHECKPOINT_DIR, checkpoint_name)
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, f=checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

            # Exit condition if we reach the target number of steps mid-epoch
            if global_step >= config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS:
                break
        
        # Another check to break the outer loop
        if global_step >= config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS:
            break

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    train_model()
    logger.info("Training script finished.")