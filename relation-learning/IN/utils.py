import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import yaml

import datagen as dg

from pathlib import Path

from tqdm import tqdm


def set_seed(seed=1213):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the config file")
    parser.add_argument(
        "--generate", action="store_true", help="Whether to generate data")

    return parser


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")

    return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                  **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]

    return nn.__getattribute__(loss_name)()


def generate_data(config: dict):
    data_size = config["size"]
    sim_type = config["type"]

    savedir = Path(config["savedir"])
    params = config["params"]

    for i in tqdm(range(data_size)):
        sim_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                value = np.random.choice(value)
            sim_params[key] = value
        sim = dg.__getattribute__(sim_type)(**sim_params)
        sim.simulate()

        sim.save(savedir=savedir, name=f"{i}.mat")


def get_device(config: dict):
    return torch.device(config["device"])


class INDataset(torchdata.Dataset):
    def __init__(self, datadir: str, sim_type: str, data_size=1000):
        self.sim_type = sim_type
        self.data_size = data_size
        self.matfiles = sorted(list(Path(datadir).glob("*.mat")))

    def __len__(self):
        return len(self.matfiles) * (self.data_size - 1)

    def __getitem__(self, idx):
        fileidx = idx // (self.data_size - 1)
        dataidx = idx % (self.data_size - 1)
        filepath = self.matfiles[fileidx]

        sim = dg.__getattribute__(self.sim_type)()
        sim.load(filepath)

        return_dict = {}

        objects = sim.objects[dataidx]
        externals = sim.externals[dataidx]
        triplets = (sim.triplets[0][dataidx], sim.triplets[1][dataidx],
                    sim.triplets[2][dataidx])

        return_dict["objects"] = objects
        return_dict["externals"] = externals
        return_dict["triplet"] = triplets

        target = sim.objects[dataidx + 1][3:5, :]
        return_dict["targets"] = target
        return return_dict


def get_loader(config: dict, phase: str):
    data_config = config["data"][phase]
    loader_config = config["loader"][phase]

    datadir = data_config["savedir"]
    sim_type = data_config["type"]
    data_size = data_config["params"]["n_steps"]

    dataset = INDataset(datadir, sim_type, data_size)
    loader = torchdata.DataLoader(dataset, **loader_config)
    return loader
