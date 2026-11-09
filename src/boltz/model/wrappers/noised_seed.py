from math import sqrt
from pathlib import Path
from typing import Any

import click
import mdtraj
import numpy as np
import torch

from .base import Boltz2Wrapper
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation
from boltz.model.loss.diffusionv2 import weighted_rigid_align


class NoisedSeedWrapper(Boltz2Wrapper):
    """Shortened reverse diffusion baseline from the reviewer.

    For each sample:
      1. Take x_init (from provided initial_structure or a Boltz-2 draw).
      2. Add noise: y = x_init + sigma_seed * eps, eps ~ N(0, I).
      3. Run reverse diffusion from sigma_seed down to 0.

    Two modes:
      - deterministic=False (default): stochastic SDE, matches the noise schedule
        used during training (gamma > 0, noise_scale > 0).
      - deterministic=True: ODE / DDIM-style, sets gamma=0 and noise_scale=0 so
        the trajectory is fully determined by the initial seed.

    This isolates whether walk-jump's gains come from Langevin exploration or
    simply from starting at an intermediate noise level with a seed structure.
    """

    def __init__(
        self,
        model: Boltz2,
        predict_args: dict[str, Any],
        steering_args: dict[str, Any],
        noised_seed_args: dict[str, Any],
        use_kernels: bool = True,
    ):
        super().__init__(model, predict_args, steering_args, use_kernels)
        self.noised_seed_args = noised_seed_args
        # noised_seed_args keys:
        #   sigma        : float  - noise level in Angstroms (match walk-jump sigma)
        #   num_samples  : int    - independent noised seeds to generate
        #   deterministic: bool  - ODE mode if True, SDE if False
        #   one_step     : bool  - single Tweedie denoise step (walk without the jump)
        # Number of multi-step denoising steps is derived automatically from sigma
        # and the model's full diffusion schedule (tail of schedule below sigma_seed).

    @torch.no_grad()
    def _denoise(
        self,
        y: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run reverse diffusion from sigma_seed down to 0 starting from y."""
        sm = self.model.structure_module
        sigma_seed = self.noised_seed_args["sigma"]
        deterministic = self.noised_seed_args.get("deterministic", False)

        sigma_data = sm.sigma_data
        step_scale = sm.step_scale
        noise_scale = 0.0 if deterministic else sm.noise_scale
        gamma_0 = 0.0 if deterministic else sm.gamma_0
        gamma_min = sm.gamma_min
        atom_mask = feats["atom_pad_mask"]

        # Use the tail of the full schedule starting from sigma_seed.
        # This ensures the step count and spacing exactly match the trained schedule.
        full_sigmas = sm.sample_schedule(self.predict_args["sampling_steps"])
        # The schedule is the actual noise level in Angstroms (sample_schedule
        # already multiplies by sigma_data), the same units as sigma_seed.
        tail_mask = full_sigmas[:-1] <= sigma_seed
        sigmas = torch.cat([full_sigmas[:-1][tail_mask], full_sigmas[1:][tail_mask]])
        # Reshape back into (num_steps + 1,) pairs by taking the tail slice directly.
        first_tail_idx = int((~tail_mask).sum().item())
        sigmas = full_sigmas[first_tail_idx:]
        num_steps = len(sigmas) - 1  # last entry is the appended 0

        click.echo(f"  Denoising with {num_steps} steps (sigma_seed={sigma_seed} Å → 0)")

        xhat = y.clone()
        xhat_denoised = None

        for i in range(num_steps):
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

            with torch.autocast("cuda", enabled=False):
                xhat_noisy = weighted_rigid_align(
                    xhat_noisy.float(),
                    xhat_denoised.float(),
                    atom_mask.float(),
                    atom_mask.float(),
                ).to(xhat_denoised)

            denoised_over_sigma = (xhat_noisy - xhat_denoised) / t_hat
            xhat = xhat_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma

        return xhat_denoised

    def process_batch(
        self,
        feats: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.move_to_device(feats)
        network_condition_kwargs = self.get_network_conditioning(feats)
        click.echo("Network conditioning obtained from Boltz2 model.")

        if feats["valid_initial_structure"].item():
            click.echo("Using provided structure as seed.")
            x_init = torch.zeros(*feats["atom_pad_mask"].shape, 3, device=self.device)
            x_init[feats["atom_pad_mask"].to(bool)] = feats["initial_structure_coords"]
        else:
            click.echo("No initial structure provided. Generating seed with Boltz2.")
            x_init = self.generate_initial_sample(feats, network_condition_kwargs)

        sigma = self.noised_seed_args["sigma"]
        num_samples = self.noised_seed_args["num_samples"]
        one_step = self.noised_seed_args.get("one_step", False)

        if one_step:
            click.echo(
                f"Generating {num_samples} samples via single-step Tweedie denoising "
                f"at sigma={sigma} Å (walk without the jump)."
            )
        else:
            mode = "ODE (deterministic)" if self.noised_seed_args.get("deterministic") else "SDE (stochastic)"
            click.echo(
                f"Generating {num_samples} samples via {mode} reverse diffusion "
                f"from sigma={sigma} Å."
            )

        samples = []
        for i in range(num_samples):
            y = x_init + sigma * torch.randn_like(x_init)
            if one_step:
                y_centered = center_random_augmentation(
                    y,
                    atom_mask=feats["atom_pad_mask"],
                    augmentation=False,
                    centering=True,
                    return_second_coords=False,
                )
                xhat = self.model.structure_module.preconditioned_network_forward(
                    y_centered,
                    sigma,
                    network_condition_kwargs=network_condition_kwargs,
                )
            else:
                xhat = self._denoise(y, feats, network_condition_kwargs)
            samples.append(xhat)

        xhat_traj = torch.stack(samples, dim=0)
        return x_init, xhat_traj

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
            sample_frequency=1,
        )
