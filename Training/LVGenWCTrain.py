

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.append("../")

from imputers.SSSDS4Imputer import SSSDS4Imputer
from utils.util import (
    calc_diffusion_hyperparams,
    find_max_epoch,
    print_size,
    training_loss_replace,
)


def load_data(train_path: str, test_path: str):
    """Load training and testing samples, keep the first two channels, and scale them."""
    train_samples = np.load(train_path)
    test_samples = np.load(test_path)

    scaler = StandardScaler().fit(train_samples.reshape(-1, train_samples.shape[-1]))

    train_scaled = scaler.transform(
        train_samples.reshape(-1, train_samples.shape[-1])
    ).reshape(train_samples.shape)

    test_scaled = scaler.transform(
        test_samples.reshape(-1, test_samples.shape[-1])
    ).reshape(test_samples.shape)

    return train_scaled, test_scaled, scaler


def load_config(config_path: str):
    """Load JSON config and extract the relevant sections."""
    with open(config_path, "r") as f:
        config = json.load(f)

    train_config = config["train_config"]
    gen_config = config["gen_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    # Fixed to SSSDS4 only
    model_config = config["wavenet_config"]

    return (
        train_config,
        gen_config,
        trainset_config,
        diffusion_config,
        diffusion_hyperparams,
        model_config,
    )


def build_model(model_config: dict, device: torch.device):
    """Build the fixed SSSDS4 model."""
    return SSSDS4Imputer(**model_config).to(device)


def prepare_output_directory(output_directory: str, diffusion_config: dict):
    """Create the experiment output directory."""
    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
    )

    full_output_directory = os.path.join(output_directory, local_path)
    os.makedirs(full_output_directory, exist_ok=True)

    print(f"Output directory: {full_output_directory}", flush=True)
    return full_output_directory


def move_diffusion_hyperparams_to_device(
    diffusion_hyperparams: dict, device: torch.device
):
    """Move diffusion hyperparameters to the selected device."""
    return {
        key: value if key == "T" else value.to(device)
        for key, value in diffusion_hyperparams.items()
    }


def load_checkpoint_if_available(net, optimizer, output_directory: str, ckpt_iter):
    """Load a checkpoint if requested and available."""
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(output_directory)

    if ckpt_iter >= 0:
        model_path = os.path.join(output_directory, f"{ckpt_iter}.pkl")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            net.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print(f"Successfully loaded model at iteration {ckpt_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("No valid checkpoint found, starting from initialization.")
            ckpt_iter = -1
    else:
        print("No valid checkpoint found, starting from initialization.")

    return ckpt_iter


def prepare_training_data(train_scaled: np.ndarray, device: torch.device):
    """
    Convert training data into a tensor with shape:
    (num_samples, sequence_length, num_features)
    """
    training_tensor = torch.from_numpy(train_scaled).float().to(device)
    print(f"Training data shape: {training_tensor.shape}")
    print("Data loaded")
    return training_tensor


def train(
    train_scaled,
    output_directory,
    ckpt_iter,
    n_iters,
    iters_per_ckpt,
    iters_per_logging,
    learning_rate,
    use_model,
    only_generate_missing,
    masking,
    missing_k,
    options,
    diffusion_config,
    diffusion_hyperparams,
    model_config,
):
    """
    Train the diffusion model.

    Notes
    -----
    - This script is fixed to SSSDS4.
    - `use_model`, `masking`, `missing_k`, and `options` are kept only for
      compatibility with the existing config structure.
    """
    del use_model, masking, missing_k, options

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    output_directory = prepare_output_directory(output_directory, diffusion_config)
    diffusion_hyperparams_device = move_diffusion_hyperparams_to_device(
        diffusion_hyperparams, device
    )

    net = build_model(model_config, device)
    print_size(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate * 0.1)

    ckpt_iter = load_checkpoint_if_available(net, optimizer, output_directory, ckpt_iter)

    training_data = prepare_training_data(train_scaled, device)
    print(f"Original training data shape: {training_data.shape}")
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            mask = torch.zeros(batch.shape, device=device).float()
            mask[:,:,2:] = 1.0  # Mask all channels except the first two
            batch = batch.permute(0, 2, 1)
            mask = mask.permute(0, 2, 1)
            loss_mask = ~mask.bool()

            assert batch.size() == mask.size() == loss_mask.size()

            optimizer.zero_grad()

            inputs = (batch, batch, mask, loss_mask)
            loss = training_loss_replace(
                net,
                nn.MSELoss(),
                inputs,
                diffusion_hyperparams_device,
                only_generate_missing=only_generate_missing,
            )

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print(f"Iteration: {n_iter}\tLoss: {loss.item()}")

            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = f"{n_iter}.pkl"
                checkpoint_path = os.path.join(output_directory, checkpoint_name)

                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"Model checkpoint saved at iteration {n_iter}")

            n_iter += 1
            if n_iter >= n_iters + 1:
                break


def main():
    train_scaled, test_scaled, scaler = load_data(
        "../data/TrainingSamples_Cleaned.npy",
        "../data/TestingSamples_Cleaned.npy",
    )

    (
        train_config,
        gen_config,
        trainset_config,
        diffusion_config,
        diffusion_hyperparams,
        model_config,
    ) = load_config("../configs/LVGenWC.json")

    train_config["options"] = [
        [144],
        [144],
    ]

    print(model_config)

    train(
        train_scaled=train_scaled,
        diffusion_config=diffusion_config,
        diffusion_hyperparams=diffusion_hyperparams,
        model_config=model_config,
        **train_config,
    )


if __name__ == "__main__":
    main()