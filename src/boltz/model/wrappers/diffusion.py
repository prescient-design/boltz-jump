from typing import Any
import torch
import click
from pathlib import Path
import numpy as np
import mdtraj
from pytorch_lightning import LightningModule
import tqdm

from .base import Boltz2Wrapper
from boltz.model.modules.integrator import BAOAB
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation


class DiffusionWrapper(Boltz2Wrapper):
    """A wrapper for Boltz2 to perform standard diffusion sampling."""

    def __init__(
        self,
        model: Boltz2,
        predict_args: dict[str, Any],
        steering_args: dict[str, Any],
        num_samples: int = 100,
        use_kernels: bool = True,
    ):
        """
        Initializes the diffusion wrapper around the Boltz2 model.

        Args:
            model (Boltz2): The pre-loaded Boltz2 model.
            predict_args (dict): Prediction arguments for the model.
            steering_args (Any): Steering arguments for the model.
            num_samples (int): Number of samples to generate per input.
            use_kernels (bool): Whether to use custom CUDA kernels.
        """
        super().__init__(model, predict_args, steering_args, use_kernels)
        self.num_samples = num_samples

    def process_batch(
        self,
        feats: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single batch of features to generate an initial sample and an walk-jump trajectory.

        Args:
            feats (dict[str, Any]): A batch of features from the dataloader.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_init (torch.Tensor): The initial coordinates, which are sampled from the model if not provided.
                - xhat_traj (torch.Tensor): The denoised trajectory coordinates.
        """
        feats = self.move_to_device(feats)

        network_condition_kwargs = self.get_network_conditioning(feats)
        click.echo("Network conditioning obtained from Boltz2 model.")

        samples = []
        for _ in tqdm.trange(self.num_samples, desc="Generating samples"):
            samples.append(self.generate_initial_sample(feats, network_condition_kwargs))
        xhat_traj = torch.stack(samples, dim=0)
        return xhat_traj[0], xhat_traj

    def write_trajectory(
        self,
        xhat_traj: torch.Tensor,
        feats: dict[str, Any],
        topology_file: str | Path,
        trajectory_file: str | Path,
    ):
        return super().write_trajectory(
            xhat_traj,
            feats,
            topology_file,
            trajectory_file,
            sample_frequency=1,
        )
