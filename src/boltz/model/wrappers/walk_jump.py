from typing import Any
import torch
import click
from pathlib import Path
import numpy as np
import mdtraj
from pytorch_lightning import LightningModule

from .base import Boltz2Wrapper
from .diversity import build_bias
from boltz.model.modules.integrator import BAOAB
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation
from boltz.model.potentials.potentials import get_potentials, GlobalStructureRMSDPotential, StructureReferencePotential


def construct_steering_potentials(steering_args):
    """Constructs steering potentials."""

    if not steering_args["structure_guidance"]:
        return get_potentials(steering_args, boltz2=True)

    if steering_args.get("structure_guidance_use_rmsd", False):
        return [
            GlobalStructureRMSDPotential(
                parameters={
                    "guidance_weight": steering_args.get(
                        "structure_guidance_weight", 0.1
                    ),
                    "guidance_interval": steering_args.get(
                        "structure_guidance_interval", 1
                    ),
                    "resampling_weight": 1.0,
                    "threshold": steering_args.get(
                        "structure_rmsd_threshold", 1.0
                    ),
                    "k": steering_args.get("structure_k", 1.0),
                    "use_rep_atoms_only": steering_args.get(
                        "structure_use_rep_atoms_only", True
                    ),
                    "check_input": steering_args.get(
                        "check_structure_input", False
                    ),
                }
            )
        ]
    else:
        return [
            StructureReferencePotential(
            parameters={
                "guidance_weight": steering_args.get(
                        "structure_guidance_weight", 0.1
                    ),
                    "guidance_interval": steering_args.get(
                        "structure_guidance_interval", 1
                    ),
                    "resampling_weight": 1.0,
                    "threshold": steering_args.get(
                        "structure_distance_threshold", 0.5
                    ),
                    "use_rep_atoms_only": steering_args.get(
                        "structure_use_rep_atoms_only", True
                    ),
                    "check_input": steering_args.get(
                        "check_structure_input", False
                    ),
                }
            )
        ]


class WalkJumpWrapper(Boltz2Wrapper):
    """A wrapper for Boltz2 to perform walk-jump sampling."""

    def __init__(
        self,
        model: Boltz2,
        predict_args: dict[str, Any],
        steering_args: dict[str, Any],
        walk_jump_args: dict[str, Any],
        use_kernels: bool = True,
    ):
        """
        Initializes the walk-jump wrapper around the Boltz2 model.

        Args:
            model (Boltz2): The pre-loaded Boltz2 model.
            predict_args (dict): Prediction arguments for the model.
            steering_args (Any): Steering arguments for the model.
            walk_jump_args (dict): Arguments specific to walk-jump sampling.
            use_kernels (bool): Whether to use custom CUDA kernels.
        """
        super().__init__(model, predict_args, steering_args, use_kernels)
        self.walk_jump_args = walk_jump_args

    @torch.no_grad()
    def walk_jump(
        self,
        x_init: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Performs walk-jump sampling starting from a clean initial structure."""
        from math import sqrt
        from boltz.model.loss.diffusionv2 import weighted_rigid_align
        
        y_init = x_init + self.walk_jump_args["sigma"] * torch.randn_like(x_init)

        def xhat_fn(y: torch.Tensor, num_denoising_steps: int) -> torch.Tensor:
            if num_denoising_steps <= 1:
                # Original single-step Tweedie denoiser
                y_centered = center_random_augmentation(
                    y,
                    atom_mask=feats["atom_pad_mask"],
                    augmentation=False,
                    centering=True,
                    return_second_coords=False,
                )
                xhat = self.model.structure_module.preconditioned_network_forward(
                    y_centered,
                    self.walk_jump_args["sigma"],
                    network_condition_kwargs=network_condition_kwargs,
                )
            else:
                # Multi-step reverse diffusion from sigma down to ~0
                sigma = self.walk_jump_args["sigma"]
                inv_rho = 1.0 / 7.0
                
                sigma_data = self.model.structure_module.sigma_data
                sigma_min = self.model.structure_module.sigma_min
                step_scale = self.model.structure_module.step_scale
                noise_scale = self.model.structure_module.noise_scale
                gamma_0 = self.model.structure_module.gamma_0
                gamma_min = self.model.structure_module.gamma_min
                
                # Karras schedule from the walk noise level down to the model's
                # minimum noise level, both in Angstroms (sigma_min is in units
                # of sigma_data, as in AtomDiffusion.sample_schedule).
                sigma_end = sigma_min * sigma_data
                steps = torch.arange(num_denoising_steps, device=y.device, dtype=torch.float32)
                sigmas = (
                    sigma ** inv_rho
                    + steps / (num_denoising_steps - 1) * (sigma_end ** inv_rho - sigma ** inv_rho)
                ) ** (1.0 / inv_rho)
                sigmas = torch.cat([sigmas, torch.zeros(1, device=y.device)])
                
                atom_mask = feats["atom_pad_mask"]
                xhat = y.clone()
                xhat_denoised = None
                
                for i in range(num_denoising_steps):
                    sigma_tm = sigmas[i].item()
                    sigma_t = sigmas[i + 1].item()
                    gamma = gamma_0 if sigma_tm > gamma_min else 0.0
                    
                    t_hat = sigma_tm * (1 + gamma)
                    noise_var = noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)
                    eps = sqrt(noise_var) * torch.randn_like(xhat) if noise_var > 0 else 0
                    xhat_noisy = xhat + eps
                    
                    xhat_centered = center_random_augmentation(
                        xhat_noisy,
                        atom_mask=atom_mask,
                        augmentation=False,
                        centering=True,
                        return_second_coords=False,
                    )
                    
                    xhat_denoised = self.model.structure_module.preconditioned_network_forward(
                        xhat_centered,
                        t_hat,
                        network_condition_kwargs=network_condition_kwargs,
                    )
                    
                    # Rigid align noisy to denoised before computing update
                    xhat_noisy = weighted_rigid_align(
                        xhat_noisy.float(),
                        xhat_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    ).to(xhat_denoised)
                    
                    denoised_over_sigma = (xhat_noisy - xhat_denoised) / t_hat
                    xhat = xhat_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
                
                xhat = xhat_denoised

            # Apply steering potentials
            steering_args = self.steering_args
            if (steering_args is None) or not (
                steering_args["fk_steering"]
                or steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
                or steering_args["structure_guidance"]
            ):
                return xhat
            
            potentials = construct_steering_potentials(steering_args)
            steering_t = 0.0
            guidance_update = torch.zeros_like(xhat)
            for guidance_step in range(steering_args["num_gd_steps"]):
                energy_gradient = torch.zeros_like(xhat)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    lambd = parameters["guidance_weight"]
                    if lambd > 0:
                        energy_gradient += lambd * \
                            potential.compute_gradient(
                                xhat + guidance_update,
                                feats,
                                parameters
                            )
                guidance_update -= energy_gradient

            xhat += guidance_update
            return xhat

        # Optional exploration bias (see diversity.py). Its gradient is added to
        # the score once per walk step, which is also when hills are deposited.
        bias = build_bias(self.walk_jump_args, x_init, feats)
        bias_mask = feats["atom_pad_mask"].to(bool)

        def score_fn(y: torch.Tensor) -> torch.Tensor:
            # Walk always uses single-step for speed
            xhat = xhat_fn(y, num_denoising_steps=1)
            score = (xhat - y) / (self.walk_jump_args["sigma"] ** 2)
            if bias is not None:
                score = score - bias.gradient(xhat, bias_mask)
                bias.maybe_deposit(xhat, bias_mask)
            return score

        mcmc = BAOAB(
            delta=self.walk_jump_args["delta"],
            friction=self.walk_jump_args["friction"],
            steps=self.walk_jump_args["steps"],
            save_every_n_steps=self.walk_jump_args["save_every_n_steps"],
            save_trajectory=True,
        )

        # Obtain 'walk' trajectory in Y space.
        _, _, y_traj, _ = mcmc(y_init, score_fn)

        # Denoise the walk trajectory — use multi-step for final output quality
        num_denoise = self.walk_jump_args.get("num_denoising_steps", 1)
        xhat_traj = torch.stack([
            xhat_fn(y_traj[i, :], num_denoising_steps=num_denoise)
            for i in range(y_traj.size(0))
        ])
        return xhat_traj, y_traj

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

        if feats["valid_initial_structure"].item():
            click.echo("Using provided structure for initializing walk-jump chain.")
            x_init = torch.zeros(*feats["atom_pad_mask"].shape, 3, device=self.device)
            x_init[feats["atom_pad_mask"].to(bool)] = feats["initial_structure_coords"]
        else:
            click.echo(
                "No initial structure provided. Generating initial structure with Boltz2."
            )
            x_init = self.generate_initial_sample(feats, network_condition_kwargs)

        xhat_traj, y_traj = self.walk_jump(x_init, feats, network_condition_kwargs)
        return x_init, xhat_traj, y_traj

    def write_trajectory(
        self,
        traj: torch.Tensor,
        feats: dict[str, Any],
        topology_file: str | Path,
        trajectory_file: str | Path,
    ):
        return super().write_trajectory(
            traj,
            feats,
            topology_file,
            trajectory_file,
            sample_frequency=self.walk_jump_args["save_every_n_steps"],
        )

