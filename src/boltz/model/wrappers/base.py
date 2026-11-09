
from typing import Any
import torch
import click
from pathlib import Path
import numpy as np
import mdtraj
from pytorch_lightning import LightningModule

from boltz.model.modules.integrator import BAOAB
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation


class Boltz2Wrapper:
    """A wrapper for Boltz2."""

    def __init__(
        self,
        model: Boltz2,
        predict_args: dict[str, Any],
        steering_args: dict[str, Any] = None,
        use_kernels: bool = True,
    ):
        """
        Initializes the base wrapper around the Boltz2 model.

        Args:
            model (Boltz2): The pre-loaded Boltz2 model.
            predict_args (dict): Prediction arguments for the model.
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.predict_args = predict_args
        self.steering_args = steering_args
        self.use_kernels = use_kernels

    def move_to_device(self, feats: dict) -> dict:
        """Moves tensors in a feature dictionary to a specified device."""
        new_feats = {}
        for k, v in feats.items():
            new_feats[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return new_feats

    @torch.no_grad()
    def get_network_conditioning(self, feats: dict[str, Any]) -> dict[str, Any]:
        """Runs the model trunk to get conditioning features for the diffusion model."""
        s_inputs = self.model.input_embedder(feats)
        s_init = self.model.s_init(s_inputs)
        z_init = (
            self.model.z_init_1(s_inputs)[:, :, None]
            + self.model.z_init_2(s_inputs)[:, None, :]
        )

        relative_position_encoding = self.model.rel_pos(feats)
        z_init += relative_position_encoding
        z_init += self.model.token_bonds(feats["token_bonds"].float())
        if self.model.bond_type_feature:
            z_init += self.model.token_bonds_type(feats["type_bonds"].long())
        z_init += self.model.contact_conditioning(feats)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for _ in range(self.predict_args["recycling_steps"] + 1):
            s = s_init + self.model.s_recycle(self.model.s_norm(s))
            z = z_init + self.model.z_recycle(self.model.z_norm(z))
            z = z + self.model.msa_module(z, s_inputs, feats, use_kernels=self.use_kernels)
            s, z = self.model.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask, use_kernels=self.use_kernels
            )

        q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
            self.model.diffusion_conditioning(
                s_trunk=s,
                z_trunk=z,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
            )
        )

        return {
            "s_trunk": s,
            "s_inputs": s_inputs,
            "feats": feats,
            "diffusion_conditioning": {
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            },
        }


    @torch.no_grad()
    def generate_initial_sample(
        self, feats: dict[str, Any], network_condition_kwargs: dict[str, Any]
    ) -> torch.Tensor:
        """Generates an initial structure using the model's diffusion sampler."""
        sample = self.model.structure_module.sample(
            feats=feats,
            atom_mask=feats["atom_pad_mask"].float(),
            s_trunk=network_condition_kwargs["s_trunk"],
            s_inputs=network_condition_kwargs["s_inputs"],
            num_sampling_steps=self.predict_args["sampling_steps"],
            multiplicity=1,
            max_parallel_samples=1,
            steering_args=self.steering_args,
            diffusion_conditioning=network_condition_kwargs["diffusion_conditioning"],
        )
        return sample["sample_atom_coords"]

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
        raise NotImplementedError

    def write_trajectory(
        self,
        xhat_traj: torch.Tensor,
        feats: dict[str, Any],
        topology_file: str | Path,
        trajectory_file: str | Path,
        sample_frequency: int = 1,
    ):
        """Writes the MCMC trajectory to a file."""
        traj = mdtraj.load(str(topology_file))

        # Remove padding and convert to nm for mdtraj.
        unpadded_coords = (
            xhat_traj[:, feats["atom_pad_mask"].to(bool), ...].cpu().numpy()
        )
        unpadded_coords_nm = unpadded_coords / 10.0  # Angstrom to nm

        traj.xyz = unpadded_coords_nm
        traj.time = np.arange(len(unpadded_coords_nm)) * sample_frequency

        traj.save_xtc(str(trajectory_file))
