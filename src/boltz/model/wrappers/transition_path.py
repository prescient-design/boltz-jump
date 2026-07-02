from typing import Any
import torch
import click
from pathlib import Path
import numpy as np
import mdtraj

from .base import Boltz2Wrapper
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation
from boltz.model.loss.diffusionv2 import weighted_rigid_align


class TransitionPathWrapper(Boltz2Wrapper):
    """
    A wrapper for Boltz2 to perform transition path sampling via 
    Onsager-Machlup action minimization.
    
    Based on: "Action-Minimization Meets Generative Modeling: Efficient 
    Transition Path Sampling with the Onsager-Machlup Functional"
    (Raja et al., 2025)
    """

    def __init__(
        self,
        model: Boltz2,
        predict_args: dict[str, Any],
        steering_args: dict[str, Any],
        om_tps_args: dict[str, Any],
        use_kernels: bool = True,
    ):
        super().__init__(model, predict_args, steering_args, use_kernels)
        self.om_tps_args = om_tps_args

    def score_fn(
        self,
        x: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute the score function s_θ(x, τ) ≈ ∇log p_τ(x).
        By Tweedie's formula: s_θ(x, τ) = (x̂_0 - x) / σ²
        """
        sigma = self.om_tps_args["sigma"]
        
        x_centered = center_random_augmentation(
            x,
            atom_mask=feats["atom_pad_mask"],
            augmentation=False,
            centering=True,
            return_second_coords=False,
        )
        
        xhat = self.model.structure_module.preconditioned_network_forward(
            x_centered,
            sigma,
            network_condition_kwargs=network_condition_kwargs,
        )
        
        score = (xhat - x_centered) / (sigma ** 2)
        return score

    def compute_path_term(self, path: torch.Tensor) -> torch.Tensor:
        """
        Term A: path smoothness.
        A[X] = (1 / 2Δt) Σᵢ ||x⁽ⁱ⁺¹⁾ - x⁽ⁱ⁾||²
        """
        delta_t = self.om_tps_args["delta_t"]
        differences = path[1:] - path[:-1]
        squared_distances = (differences ** 2).sum(dim=(-1, -2))
        return squared_distances.sum() / (2 * delta_t)

    def compute_drift_term(
        self,
        path: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Term B: drift norm.
        B[X] = (Δt / 2ζ²) Σᵢ ||s_θ(x⁽ⁱ⁾, τ)||²
        
        Processes one point at a time (like walk-jump) to avoid batching issues
        with network conditioning.
        """
        delta_t = self.om_tps_args["delta_t"]
        friction = self.om_tps_args["friction"]
        sigma = self.om_tps_args["sigma"]
        
        drift_term = 0.0
        
        # Process interior points one at a time
        for i in range(1, path.shape[0] - 1):
            x_i = path[i]  # [batch_size, num_atoms, 3]
            
            x_centered = center_random_augmentation(
                x_i,
                atom_mask=feats["atom_pad_mask"],
                augmentation=False,
                centering=True,
                return_second_coords=False,
            )
            
            xhat = self.model.structure_module.preconditioned_network_forward(
                x_centered,
                sigma,
                network_condition_kwargs=network_condition_kwargs,
            )
            
            # Score = (xhat - x) / σ²
            score = (xhat - x_centered) / (sigma ** 2)
            score_norm_sq = (score ** 2).sum()
            drift_term = drift_term + score_norm_sq
        
        return drift_term * delta_t / (2 * friction ** 2)

    def compute_divergence_term(
        self,
        path: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Term C: divergence using Hutchinson trace estimator.
        C[X] = (D·Δt / ζ) Σᵢ ∇·s_θ(x⁽ⁱ⁾, τ)
        """
        delta_t = self.om_tps_args["delta_t"]
        friction = self.om_tps_args["friction"]
        diffusivity = self.om_tps_args["diffusivity"]
        n_samples = self.om_tps_args["hutchinson_samples"]
        
        divergence_term = 0.0
        
        for i in range(1, path.shape[0] - 1):
            x_i = path[i].requires_grad_(True)
            score = self.score_fn(x_i, feats, network_condition_kwargs)
            
            trace_estimate = 0.0
            for _ in range(n_samples):
                v = torch.randn_like(score)
                vjp = torch.autograd.grad(
                    outputs=score,
                    inputs=x_i,
                    grad_outputs=v,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                trace_estimate = trace_estimate + (v * vjp).sum()
            
            divergence_term = divergence_term + trace_estimate / n_samples
        
        return divergence_term * diffusivity * delta_t / friction

    def compute_om_action(
        self,
        path: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the Onsager-Machlup action functional.
        
        Full:      S[X] = (1/2D) (A[X] + B[X] + C[X])
        Truncated: S[X] = (1/2D) (A[X] + B[X])
        """
        diffusivity = self.om_tps_args["diffusivity"]
        action_type = self.om_tps_args["action_type"]
        
        path_term = self.compute_path_term(path)
        drift_term = self.compute_drift_term(path, feats, network_condition_kwargs)
        
        if action_type == "full" and self.om_tps_args["use_divergence"]:
            divergence_term = self.compute_divergence_term(path, feats, network_condition_kwargs)
        else:
            divergence_term = torch.tensor(0.0, device=path.device)
        
        action = (path_term + drift_term + divergence_term) / (2 * diffusivity)
        
        return action, {
            "path_term": path_term.item(),
            "drift_term": drift_term.item(),
            "divergence_term": divergence_term.item() if torch.is_tensor(divergence_term) else 0.0,
        }

    def generate_initial_path(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Generate initial guess path via latent interpolation.
        
        1. Encode endpoints by adding noise at level sigma
        2. Linearly interpolate in the noised space
        3. Decode each point by denoising (single-step, like walk-jump)
        """
        L = self.om_tps_args["path_length"]
        sigma = self.om_tps_args["sigma"]
        
        # Encode endpoints by adding noise
        y_start = x_start + sigma * torch.randn_like(x_start)
        y_end = x_end + sigma * torch.randn_like(x_end)
        
        # Linear interpolation in noised space
        alphas = torch.linspace(0, 1, L, device=x_start.device)
        y_path = torch.stack([
            (1 - alpha) * y_start + alpha * y_end
            for alpha in alphas
        ])  # [L, batch_size, num_atoms, 3]
        
        return y_path

    def optimize_path(
        self,
        path: torch.Tensor,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Optimize the path by minimizing the OM action.
        """
        optimization_steps = self.om_tps_args["optimization_steps"]
        learning_rate = self.om_tps_args["learning_rate"]
        
        # Only optimize interior points
        interior_path = path[1:-1].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([interior_path], lr=learning_rate)
        
        history = []
        
        for step in range(optimization_steps):
            optimizer.zero_grad()
            
            full_path = torch.cat([
                path[0:1].detach(),
                interior_path,
                path[-1:].detach()
            ], dim=0)
            
            action, terms = self.compute_om_action(full_path, feats, network_condition_kwargs)
            action.backward()
            optimizer.step()
            
            click.echo(f"  Step {step}: Action={action.item():.4f}, "
                        f"Path Term={terms['path_term']:.4f}, Drift Term={terms['drift_term']:.4f}")
            
            if step % 10 == 0:
                decoded_path = self.decode_path(
                    full_path, x_start=x_start, x_end=x_end, feats=feats,
                    network_condition_kwargs=network_condition_kwargs
                )
                history.append(
                    {"step": step, "action": action.item(), "path": decoded_path.detach().cpu(), **terms}
                )
        
        optimized_path = torch.cat([
            path[0:1].detach(),
            interior_path.detach(),
            path[-1:].detach()
        ], dim=0)
        
        return optimized_path, history

    def decode_path(
        self,
        path: torch.Tensor,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        feats: dict[str, Any],
        network_condition_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Decode a path from noisy space to clean space.
        """
        L = path.shape[0]
        sigma = self.om_tps_args["sigma"]
        
        decoded_path = []
        for i in range(L):
            y = path[i]
            
            y_centered = center_random_augmentation(
                y,
                atom_mask=feats["atom_pad_mask"],
                augmentation=False,
                centering=True,
                return_second_coords=False,
            )
            
            # Single-step denoise (like walk-jump's xhat_fn)
            x_i = self.model.structure_module.preconditioned_network_forward(
                y_centered,
                sigma,
                network_condition_kwargs=network_condition_kwargs,
            )
            decoded_path.append(x_i)
        
        decoded_path = torch.stack(decoded_path)  # [L, batch_size, num_atoms, 3]        
        
        # Fix endpoints exactly
        decoded_path[0] = x_start
        decoded_path[-1] = x_end

        return decoded_path

    def process_batch(
        self,
        feats: dict[str, Any],
    ) -> tuple[torch.Tensor, list[dict]]:
        """Process a batch to generate transition path."""
        feats = self.move_to_device(feats)
        network_condition_kwargs = self.get_network_conditioning(feats)
        click.echo("Network conditioning obtained.")

        # Get start and end structures directly from feats
        x_start = torch.zeros(*feats["atom_pad_mask"].shape, 3, device=self.device)
        x_start[feats["atom_pad_mask"].to(bool)] = feats["initial_structure_coords"]
        x_end = torch.zeros(*feats["atom_pad_mask"].shape, 3, device=self.device)
        x_end[feats["atom_pad_mask"].to(bool)] = feats["final_structure_coords"]
        
        # Align start to end
        x_start = weighted_rigid_align(
            x_start,
            x_end,
            feats["atom_pad_mask"],
            feats["atom_pad_mask"],
        )

        click.echo("Generating initial path via latent interpolation...")
        with torch.no_grad():
            initial_path = self.generate_initial_path(
                x_start, x_end, feats, network_condition_kwargs
            )

        click.echo(f"Optimizing path with {initial_path.shape[0]} points...")
        with torch.enable_grad():
            optimized_path, history = self.optimize_path(
                initial_path, x_start, x_end, feats, network_condition_kwargs
            )
        
        with torch.no_grad():
            decoded_path = self.decode_path(
                optimized_path, x_start, x_end, feats, network_condition_kwargs
            )

        return decoded_path, history