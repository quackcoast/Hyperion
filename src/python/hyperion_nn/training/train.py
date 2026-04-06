# this will be used for the following:
# - this is the main training script that will be used to train the neural network
# - load config, initialize the Dataset andDataLoader, and initialize the model
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
from torch.utils.data import DataLoader, Subset, ConcatDataset
import hyperion_nn.utils.constants as constants


# 1. Get the root logger. This is the master logger that all others inherit from.

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 2. Create the file handler. This handler writes messages to log files.
log_file_path = os.path.join(config.PathsConfig.LOGS_DIR, "training.log")
file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
file_handler.setLevel(logging.INFO) # Log everything of level INFO and above to the file.
file_formatter = logging.Formatter("%(asctime)s [%(name)-30s] [%(levelname)-8s] %(message)s")
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# 3. Create the console handler. This handler prints messages to the console
console_handler = logging.StreamHandler(sys.stdout)
# Set a higher level for the console. WARNING means it will only print messages
# that are warnings, errors, or critical. INFO messages will be ignored.
console_handler.setLevel(logging.DEBUG) 
console_formatter = logging.Formatter("[%(levelname)-8s] %(message)s")
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def collate_fn(batch):
        """
        Filters out None values from a batch.
        These None values are returned by __getitem__ when it encounters a bad row.
        """
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

def validate_model(model, validation_loader, device, policy_loss_fn, value_loss_fn, global_step):
    """
    Validates the model and saves raw prediction data for later analysis.
    """
    logger.info(f"Starting validation at global step {global_step}...")
    model.eval()

    # initialize accumulators
    total_policy_loss, total_value_loss, total_loss = 0, 0, 0
    correct_policy_predictions, total_policy_predictions = 0, 0
    correct_value_predictions, total_value_predictions = 0, 0
    processed_batches = 0
    
    # lists to store raw data for plotting
    all_value_predictions = []
    all_value_targets = []

    # paths
    validation_output_file = os.path.join(config.PathsConfig.POST_VALIDATION_DATA_DIR, "validation.txt")

    with torch.no_grad():
        progress_bar = tqdm(validation_loader, desc=f"Validation at Step {global_step}", leave=False)
        for batch in progress_bar:
            if batch is None:
                continue

            processed_batches += 1

            input_planes, policy_target, value_target = batch
            input_planes = input_planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)

            # use AMP autocast for validation too (faster inference)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                policy_logits, value_output = model(input_planes)
                value_prediction = torch.tanh(value_output)

                # loss calculation
                loss_policy = policy_loss_fn(policy_logits, torch.argmax(policy_target, dim=1))
                loss_value = value_loss_fn(value_prediction, value_target)

            # store raw predictions and targets for later plotting
            all_value_predictions.append(value_prediction.cpu())
            all_value_targets.append(value_target.cpu())

            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            total_loss += (loss_policy + loss_value).item()

            # accuracy calculation (for quick summary)
            _, predicted_policy_indices = torch.max(policy_logits, 1)
            _, target_policy_indices = torch.max(policy_target, 1)
            correct_policy_predictions += (predicted_policy_indices == target_policy_indices).sum().item()
            total_policy_predictions += policy_target.size(0)

            predicted_value_outcome = torch.round(value_prediction.squeeze()).long()
            target_value_outcome = value_target.squeeze().long()
            correct_value_predictions += (predicted_value_outcome == target_value_outcome).sum().item()
            total_value_predictions += value_target.size(0)

    # save raw results to a file
    all_value_predictions = torch.cat(all_value_predictions)
    all_value_targets = torch.cat(all_value_targets)
    results_path = os.path.join(config.PathsConfig.POST_VALIDATION_DATA_DIR, f"validation_results_step_{global_step}.pt")
    torch.save({'predictions': all_value_predictions, 'targets': all_value_targets}, results_path)
    logger.info(f"Saved raw validation results for plotting to {results_path}")

    # log summary
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_policy_loss = total_policy_loss / processed_batches
        avg_value_loss = total_value_loss / processed_batches
        policy_acc = (correct_policy_predictions / total_policy_predictions) * 100
        value_acc = (correct_value_predictions / total_value_predictions) * 100
        log_message = (f"Validation Summary at Step {global_step}: "
                       f"Avg Loss: {avg_loss:.4f} (P: {avg_policy_loss:.4f} V: {avg_value_loss:.4f}) | "
                       f"Policy Acc: {policy_acc:.2f}% | "
                       f"Value Acc (Rounded): {value_acc:.2f}%\n")
        with open(validation_output_file, 'a') as f:
            f.write(log_message)
        logger.info(log_message.strip())

    model.train()
    logger.info("Validation finished.")

def worker_init_fn(worker_id):
    """
    Called on each worker process initialization.
    !! Must call the init_worker() method on every subclass of Dataset that uses LMDB. !!
    """

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            ds.init_worker()
    else:
        dataset.init_worker()

def _create_dataset(raw_path, processed_path) -> ConcatDataset:
    """
    Helper function to create a ChessDataset instance.
    Either raw_path or processed_path must be provided.
    """
    csv_files = glob.glob(os.path.join(raw_path, "*.csv"))
    lmdb_files = [f for f in glob.glob(os.path.join(processed_path, "*.lmdb")) 
                  if not f.endswith('.lmdb-lock')]

    csv_file_names_set = set([os.path.splitext(os.path.basename(file))[0] for file in csv_files])
    lmdb_file_names_set = set([os.path.splitext(os.path.basename(file))[0] for file in lmdb_files])
    logger.info(f"CSV files found: {csv_file_names_set} LMDB files found: {lmdb_file_names_set}")
    logger.info(f"CSV not set: {csv_files} LMDB not set: {lmdb_files}")

    csvs_to_shard = csv_file_names_set - lmdb_file_names_set
    csvs_to_shard = [os.path.join(raw_path, f"{csv_name}.csv") for csv_name in csvs_to_shard]
    logger.info(f"Missing LMDB files for CSVs: {csvs_to_shard}")

    datasets = []
    for csv_path in csvs_to_shard:
        logger.info(f"Missing LMDB shard for {csv_path}, creating it now...")
        dataset = ChessDataset(csv_file_path=csv_path)
        datasets.append(dataset)

    for lmdb_path in lmdb_files:
        dataset = ChessDataset(lmdb_file_path=lmdb_path)
        datasets.append(dataset)

    # 4) concatenate datasets and create dataloader
    dataset = ConcatDataset(datasets)

    return dataset

def train_model():
    '''Main function to train the HyperionNN model.'''

    logger.info("Starting training process...")
    device = config.HardwareBasedConfig.DEVICE
    logger.info(f"Using device: {device}")

    # 0) create necessary directories
    os.makedirs(config.PathsConfig.DATA_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.MODELS_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.RAW_TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.RAW_VALIDATION_DATA_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.PROCESSED_TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.PROCESSED_VALIDATION_DATA_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.STEPS_LOG_DIR, exist_ok=True)
    os.makedirs(config.PathsConfig.POST_VALIDATION_DATA_DIR, exist_ok=True)

    torch.set_float32_matmul_precision('high')

    logger.info("Initializing model and optimizer...")

    model = HyperionNN().to(device)

    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.ModelConfig.NUM_RESIDUAL_BLOCKS}b-{config.ModelConfig.NUM_FILTERS}f | "
                f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    if sys.platform == "win32":
        logger.info("Skipping torch.compile() — Triton (required by the Inductor backend) is not supported on Windows.")
    else:
        try:
            model = torch.compile(model)
            logger.info("Model compiled successfully with torch.compile().")
        except Exception as e:
            logger.warning(f"torch.compile() failed with error: {e}. Proceeding with the un-compiled model.")

    # AdamW decouples weight decay from the adaptive learning rate, which generally
    # leads to better generalization than vanilla Adam with weight_decay.
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=config.TrainingConfig.LEARNING_RATE,
                            weight_decay=config.TrainingConfig.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 2) checkpoint loading
    global_step = 0
    start_epoch = 0
    checkpoint_dir = config.PathsConfig.CHECKPOINT_DIR
    if os.path.exists(checkpoint_dir):
        checkpoints_list = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
        if checkpoints_list:
            latest_checkpoint_path = max(checkpoints_list, key=os.path.getctime)
            logger.info(f"Loading checkpoint from {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            start_epoch = checkpoint.get('epoch', 0)
            # Restore scaler state if available
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Checkpoint loaded. Resuming from step {global_step} (Epoch {start_epoch}).")
        else:
            logger.warning("No checkpoints found. Starting training from scratch.")

    # 3) load data

    training_dataset = _create_dataset(
        raw_path=config.PathsConfig.RAW_TRAINING_DATA_DIR,
        processed_path=config.PathsConfig.PROCESSED_TRAINING_DATA_DIR
    )

    use_persistent = config.HardwareBasedConfig.NUM_WORKERS > 0

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=config.HardwareBasedConfig.BATCH_SIZE,
        shuffle=True, 
        num_workers=config.HardwareBasedConfig.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,  
        persistent_workers=use_persistent,
    )

    # Setup validation dataloader

   
    logger.info(f"Loading validation dataset from {config.PathsConfig.RAW_VALIDATION_DATA_DIR} and {config.PathsConfig.PROCESSED_VALIDATION_DATA_DIR}...")
    validation_dataset = _create_dataset(
        raw_path=config.PathsConfig.RAW_VALIDATION_DATA_DIR,
        processed_path=config.PathsConfig.PROCESSED_VALIDATION_DATA_DIR
    )

    # Take the first 32768 positions for validation as requested
    num_validation_samples = min(32768, len(validation_dataset))
    validation_indices = list(range(num_validation_samples))
    validation_subset = Subset(validation_dataset, validation_indices)

    validation_dataloader = DataLoader(
        dataset=validation_subset,
        batch_size=config.HardwareBasedConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=config.HardwareBasedConfig.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=use_persistent,
    )
    if len(validation_subset) > 0:
        logger.info(f"Validation dataset successfully loaded with {len(validation_subset)} samples from {config.PathsConfig.PROCESSED_VALIDATION_DATA_DIR}.")
    else:
        validation_dataloader = None
        logger.warning(f"No validation data was found. Validation will be skipped.")


    # 4) init loss functions
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # 5) Learning rate scheduler — OneCycleLR with warmup + cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.TrainingConfig.LEARNING_RATE,
        total_steps=config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS,
        pct_start=config.TrainingConfig.LR_WARMUP_PCT,
    )
    # Fast-forward scheduler to current global_step if resuming
    if global_step > 0:
        logger.info(f"Fast-forwarding LR scheduler to step {global_step}...")
        for _ in range(global_step):
            scheduler.step()

    # 6) main training loop
    logger.info(f"Starting training loop at step {global_step}...")
    logger.info(f"Config: batch_size={config.HardwareBasedConfig.BATCH_SIZE}, "
                f"lr={config.TrainingConfig.LEARNING_RATE}, "
                f"weight_decay={config.TrainingConfig.WEIGHT_DECAY}, "
                f"grad_clip={config.TrainingConfig.MAX_GRAD_NORM}, "
                f"AMP={'enabled' if device.type == 'cuda' else 'disabled'}")
    model.train()

    steps_per_epoch = len(training_dataloader)
    total_epochs = (config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS // steps_per_epoch) + 1

    print(constants.TRAINING_HEADER_ART)

    for epoch in range(start_epoch, total_epochs):
        progress_bar = tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)

        for batch in progress_bar:
            if batch is None:
                logger.warning("Skipping an entire batch due to malformed data.")
                continue
            input_planes, policy_target, value_target = batch

            input_planes = input_planes.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                policy_logits, value_output = model(input_planes)

                # calculate losses
                policy_loss_indices = torch.argmax(policy_target, dim=1)
                loss_policy = policy_loss_fn(policy_logits, policy_loss_indices)
                value_prediction = torch.tanh(value_output)
                loss_value = value_loss_fn(value_prediction, value_target)
                total_loss = loss_policy + config.TrainingConfig.VALUE_LOSS_WEIGHT * loss_value

            scaler.scale(total_loss).backward()

            # Gradient clipping — stabilizes training when loss is large/volatile
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.TrainingConfig.MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            # Step the LR scheduler
            scheduler.step()

            global_step += 1

            # Update the progress bar with the latest loss information
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                p_loss=f"{loss_policy.item():.4f}",
                v_loss=f"{loss_value.item():.4f}",
                lr=f"{current_lr:.2e}",
                step=global_step,
                pos_per_sec=f"{(progress_bar.format_dict.get('rate', 0) or 0) * config.HardwareBasedConfig.BATCH_SIZE:.0f}")

            if global_step % config.TrainingConfig.LOG_EVERY_N_STEPS == 0:
                log_message = (f"Step [{global_step}/{config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS}], "
                               f"Loss: {total_loss.item():.4f} (P: {loss_policy.item():.4f} V: {loss_value.item():.4f}), "
                               f"LR: {current_lr:.2e}")
                with open(os.path.join(config.PathsConfig.STEPS_LOG_DIR, "step_log.txt"), 'a') as f:
                    f.write(log_message + '\n')

            # validation logic - this is currently broken "AttributeError: 'Subset' object has no attribute 'init_worker'"
            if global_step % config.TrainingConfig.VALIDATE_EVERY_N_STEPS == 0:
                if validation_dataloader:
                    validate_model(model, validation_dataloader, device, policy_loss_fn, value_loss_fn, global_step)
                else:
                    logger.info("Skipping validation as validation_dataloader is not available.")
                model.train()


            if global_step % config.TrainingConfig.SAVE_CHECKPOINTS_EVERY_N_STEPS == 0:
                checkpoint_name = f"checkpoint_step_{global_step}_{config.ModelConfig.NUM_RESIDUAL_BLOCKS}B-{config.ModelConfig.NUM_FILTERS}F.pt"
                checkpoint_path = os.path.join(config.PathsConfig.CHECKPOINT_DIR, checkpoint_name)
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': total_loss.item(),
                }, f=checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

            # exit condition if we reach the target number of steps mid-epoch
            if global_step >= config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS:
                break

        # another check to break the outer loop
        if global_step >= config.TrainingConfig.TOTAL_TARGET_TRAINING_STEPS:
            break

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    train_model()
    logger.info("Training script finished.")