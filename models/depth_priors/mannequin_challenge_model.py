#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import wget
from zipfile import ZipFile
import torch
import torch.autograd as autograd

from .mannequin_challenge.models import pix2pix_model
from .mannequin_challenge.options.train_options import TrainOptions
from .depth_model import DepthModel

def get_model_from_url(url: str, local_path: str, is_zip: bool = False, path_root: str = "checkpoints") -> str:
    local_path = os.path.join(path_root, local_path)
    if os.path.exists(local_path):
        print(f"Found cache {local_path}")
        return local_path

    # download
    local_path = local_path.rstrip(os.sep)
    download_path = local_path if not is_zip else f"{local_path}.zip"
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if os.path.isfile(download_path):
        print(f"Found cache {download_path}")
    else:
        print(f"Dowloading {url} to {download_path} ...")
        wget.download(url, download_path)

    if is_zip:
        print(f"Unziping {download_path} to {local_path}")
        with ZipFile(download_path, 'r') as f:
            f.extractall(local_path)
        os.remove(download_path)

    return local_path

class SuppressedStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class MannequinChallengeModel(DepthModel):
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self):
        super().__init__()

        parser = TrainOptions()
        parser.initialize()
        params = parser.parser.parse_args(["--input", "single_view"])
        params.isTrain = False

        model_file = get_model_from_url(
            "https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth",
            "mc.pth", path_root=os.path.dirname(__file__)
        )

        class FixedMcModel(pix2pix_model.Pix2PixModel):
            # Override the load function, so we can load the snapshot stored
            # in our specific location.
            def load_network(self, network, network_label, epoch_label):
                return torch.load(model_file)

        with SuppressedStdout():
            self.model = FixedMcModel(params)

    def train(self):
        self.model.switch_to_train()

    def eval(self):
        self.model.switch_to_eval()

    def parameters(self):
        return self.model.netG.parameters()

    def estimate_depth(self, images):
        images = autograd.Variable(images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)

        self.model.prediction_d, _ = self.model.netG.forward(images)

        # Reshape X1HW -> BNHW
        out_shape = shape[:-3] + self.model.prediction_d.shape[-2:]
        self.model.prediction_d = self.model.prediction_d.reshape(out_shape)

        self.model.prediction_d = torch.exp(self.model.prediction_d)
        self.model.prediction_d = self.model.prediction_d.squeeze(-3)

        return self.model.prediction_d

    def save(self, file_name):
        state_dict = self.model.netG.state_dict()
        torch.save(state_dict, file_name)
