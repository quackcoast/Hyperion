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
import itertools

import hyperion_nn.config as config
from hyperion_nn.data_utils.dataset import ChessDataset
from hyperion_nn.models.resnet_cnn import HyperionNN
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(name)-30s] [%(levelname)-8s] %(message)s",
                    handlers=[logging.FileHandler(os.path.join(config.PathsConfig.LOGS_DIR, "training.log")), logging.StreamHandler(sys.stdout)])

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
            logger.info(f"Checkpoint loaded. Resuming from step {global_step}.")
        else:
            logger.warning("No checkpoints found. Starting training from scratch.")

    # 3) load data
    training_dataset = ChessDataset()
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config.HardwareBasedConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=config.HardwareBasedConfig.NUM_WORKERS,
        pin_memory=True, # Pin memory for faster data transfer to GPU
    )
    data_iterator = iter(itertools.cycle(training_dataloader))  # Cycle through the dataset indefinitely

    # 4) init loss functions
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # 5) main training loop
    logger.info(f"Starting training loop at step {global_step}...")
    model.train()
    while global_step < config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS:
        try:
            input_planes, policy_target, value_target = next(data_iterator)
        except StopIteration:
            logger.info("End of dataset reached. Restarting from the beginning.")
            data_iterator = iter(itertools.cycle(training_dataloader))
            input_planes, policy_target, value_target = next(data_iterator)
            continue
        
        input_planes = input_planes.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        
        optimizer.zero_grad()

        policy_logits, value_output = model(input_planes)

        # calculate losses
        policy_loss_indices = torch.argmax(policy_target, dim=1)
        loss_policy = policy_loss_fn(policy_logits, policy_loss_indices)

        value_prediction = torch.tanh(value_output)
        loss_value = value_loss_fn(value_prediction, value_target.unsqueeze(1))

        total_loss = loss_policy + loss_value

        # backpropagation
        total_loss.backward()
        optimizer.step()
        global_step += 1

        # logging
        if global_step % config.TrainingConfig.LOG_EVERY_N_STEPS == 0:
            print(f"Step [{global_step}/{config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS}], Loss: {total_loss.item():.4f}")
        
        if global_step % config.VALIDATE_EVERY_N_STEPS == 0:
            # TODO: Implement the validation logic here
            # validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
            # print(f"--- Validation at Step {global_step}, Val Loss: {validation_loss:.4f} ---")
            model.train() # Switch back to training mode after evaluation

        if global_step % config.SAVE_CHECKPOINT_EVERY_N_STEPS == 0:
            checkpoint_name = f"checkpoint_step_{global_step}.pt"
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }, f=os.path.join(config.CHECKPOINT_PATH, checkpoint_name))
            print(f"Checkpoint saved at step {global_step}")

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    train_model()
    logger.info("Training script finished.")




