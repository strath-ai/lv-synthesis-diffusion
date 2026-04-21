import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from imputers.SSSDS4Imputer import SSSDS4Imputer
from utils.util import (
    calc_diffusion_hyperparams,
    find_max_epoch,
    print_size,
    sampling,
)


def load_config():
    with open(os.path.join(BASE_DIR, "configs", "LVGenWCS.json"), "r") as f:
        config = json.load(f)

    train_config = config["train_config"]
    gen_config = config["gen_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    model_config = config["wavenet_config"]

    return (
        train_config,
        gen_config,
        trainset_config,
        diffusion_config,
        diffusion_hyperparams,
        model_config,
    )


def load_data():
    train_samples = np.load(os.path.join(BASE_DIR, "data", "TrainingSamples_Cleaned.npy"))
    test_samples = np.load(os.path.join(BASE_DIR, "data", "TestingSamples_Cleaned.npy"))

    scaler = StandardScaler().fit(train_samples.reshape(-1, train_samples.shape[-1]))

    train_scaled = scaler.transform(
        train_samples.reshape(-1, train_samples.shape[-1])
    ).reshape(train_samples.shape)

    test_scaled = scaler.transform(
        test_samples.reshape(-1, test_samples.shape[-1])
    ).reshape(test_samples.shape)

    # Stats for channel 0
    train_channel_0 = train_scaled[:, :, 0:1]
    test_channel_0 = test_scaled[:, :, 0:1]

    min_train = np.amin(train_channel_0, axis=1, keepdims=True)
    mean_train = np.mean(train_channel_0, axis=1, keepdims=True)
    max_train = np.amax(train_channel_0, axis=1, keepdims=True)

    min_test = np.amin(test_channel_0, axis=1, keepdims=True)
    mean_test = np.mean(test_channel_0, axis=1, keepdims=True)
    max_test = np.amax(test_channel_0, axis=1, keepdims=True)

    min_train = np.repeat(min_train, 144, axis=1)
    mean_train = np.repeat(mean_train, 144, axis=1)
    max_train = np.repeat(max_train, 144, axis=1)

    min_test = np.repeat(min_test, 144, axis=1)
    mean_test = np.repeat(mean_test, 144, axis=1)
    max_test = np.repeat(max_test, 144, axis=1)

    train_scaled = np.concatenate((train_scaled, min_train, mean_train, max_train), axis=2)
    test_scaled = np.concatenate((test_scaled, min_test, mean_test, max_test), axis=2)

    # Stats for channel 1
    train_channel_1 = train_scaled[:, :, 1:2]
    test_channel_1 = test_scaled[:, :, 1:2]

    min_train = np.amin(train_channel_1, axis=1, keepdims=True)
    mean_train = np.mean(train_channel_1, axis=1, keepdims=True)
    max_train = np.amax(train_channel_1, axis=1, keepdims=True)

    min_test = np.amin(test_channel_1, axis=1, keepdims=True)
    mean_test = np.mean(test_channel_1, axis=1, keepdims=True)
    max_test = np.amax(test_channel_1, axis=1, keepdims=True)

    min_train = np.repeat(min_train, 144, axis=1)
    mean_train = np.repeat(mean_train, 144, axis=1)
    max_train = np.repeat(max_train, 144, axis=1)

    min_test = np.repeat(min_test, 144, axis=1)
    mean_test = np.repeat(mean_test, 144, axis=1)
    max_test = np.repeat(max_test, 144, axis=1)

    train_scaled = np.concatenate((train_scaled, min_train, mean_train, max_train), axis=2)
    test_scaled = np.concatenate((test_scaled, min_test, mean_test, max_test), axis=2)

    return train_scaled, test_scaled, scaler


def prepare_output_directory(output_directory, diffusion_config):
    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
    )

    full_output_directory = os.path.join(output_directory, local_path)
    os.makedirs(full_output_directory, exist_ok=True)
    print("Output directory:", full_output_directory)

    return full_output_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_iter", default="max")
    parser.add_argument("--output_directory", type=str, default=os.path.join(BASE_DIR, "Inference", "results", "LVGenWCS"))
    parser.add_argument("--ckpt_path", type=str, default=os.path.join(BASE_DIR, "Training", "results", "LVGenWCS"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_config,
        gen_config,
        trainset_config,
        diffusion_config,
        diffusion_hyperparams,
        model_config,
    ) = load_config()

    train_scaled, test_scaled, scaler = load_data()

    output_directory = prepare_output_directory(args.output_directory, diffusion_config)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    net = SSSDS4Imputer(**model_config).to(device)
    print_size(net)

    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
    )

    ckpt_dir = os.path.join(args.ckpt_path, local_path)

    ckpt_iter = args.ckpt_iter
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(ckpt_dir)

    model_path = os.path.join(ckpt_dir, f"{ckpt_iter}.pkl")
    print("Loading checkpoint from:", model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded model at iteration {ckpt_iter}")

    testing_data = np.split(test_scaled, 5, 0)
    testing_data = np.array(testing_data)
    testing_data = torch.from_numpy(testing_data).float().to(device)

    print("Testing data shape:", testing_data.shape)

    all_mse = []

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(testing_data):
            batch = batch.to(device).float()
            mask = torch.zeros(batch.shape, device=device).float()
            mask[:, :, 2:] = 1.0

            batch = batch.permute(0, 2, 1)
            mask = mask.permute(0, 2, 1)

            sample_length = batch.size(2)
            sample_channels = batch.size(1)

            generated_audio = sampling(
                net,
                (batch.size(0), sample_channels, sample_length),
                diffusion_hyperparams,
                cond=batch,
                mask=mask,
                only_generate_missing=1,
            )

            generated_audio = generated_audio.detach().cpu().numpy()
            batch_np = batch.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            np.save(os.path.join(output_directory, f"imputation{i}.npy"), generated_audio)
            np.save(os.path.join(output_directory, f"original{i}.npy"), batch_np)
            np.save(os.path.join(output_directory, f"mask{i}.npy"), mask_np)

            print(f"Saved generated samples for batch {i}")

            mse = mean_squared_error(
                generated_audio[~mask_np.astype(bool)],
                batch_np[~mask_np.astype(bool)],
            )
            all_mse.append(mse)

    print("Total MSE:", np.mean(all_mse))


if __name__ == "__main__":
    main()