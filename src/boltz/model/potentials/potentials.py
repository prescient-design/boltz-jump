from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, List, Union

import torch
import numpy as np
from boltz.data import const
from boltz.model.potentials.schedules import (
    ParameterSchedule,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.model.loss.diffusionv2 import weighted_rigid_align


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1)

        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, dEnergy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=True
        )
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            f = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                "sum",
            )
            dSoftmax = (
                dEnergy
                * softmax_energy
                * (1 + parameters["union_lambda"] * (energy - f[..., union_index]))
            )
            prod = dSoftmax.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),
                dSoftmax.tile(grad_value.shape[-3]).unsqueeze(-1)
                * grad_value.flatten(start_dim=-3, end_dim=-2),
                "sum",
            )
        else:
            prod = dEnergy.tile(grad_value.shape[-3]).unsqueeze(
                -1
            ) * grad_value.flatten(start_dim=-3, end_dim=-2)
            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),  # 9 x 516 x 3
                prod,
                "sum",
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., ref_token_index, :]

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        return parameters

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    def get_reference_coords(self, feats, parameters):
        return None, None


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.expand_as(value).clone()
        upper_bounds = upper_bounds.expand_as(value).clone()

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[
                ~unbounded_above_mask * ~negation_mask
            ]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float("inf")
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[
                ~unbounded_below_mask * ~negation_mask
            ]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float("-inf")

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )

        r = coords[:, index] - aligned_ref_coords
        r_norm = torch.linalg.norm(r, dim=-1)

        # print(f"r_norm shape: {r_norm.shape}, ref_mask shape: {ref_mask.shape}")
        # print(f"r_norm mean: {r_norm.mean():.4f}, max: {r_norm.max():.4f}")
        # print(f"ref_coords nonzero: {(ref_coords.abs().sum(-1) > 0).sum()}")
        # print(f"aligned_ref_coords range: {aligned_ref_coords[0, 0, ref_mask[0]].min():.2f} to {aligned_ref_coords[0, 0, ref_mask[0]].max():.2f}")
        # print(f"coords range: {coords[0, index[0]].min():.2f} to {coords[0, index[0]].max():.2f}")

        if not compute_gradient:
            return r_norm

        r_hat = r / r_norm.unsqueeze(-1)
        grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return r_norm, grad


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / (r_kj_norm**2)
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm**2).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm**2).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=pair_index.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=pair_index.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(
            lower_bounds[~bond_mask], bond_cutoffs[~bond_mask]
        )
        upper_bounds[bond_mask] = torch.min(
            upper_bounds[bond_mask], bond_cutoffs[bond_mask]
        )

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        # print("template_force", "template_force" in feats)
        # print("template_mask_cb", "template_mask_cb" in feats)
        # raise ValueError(("template_force" in feats, "template_mask_cb" in feats))
        if self.parameters["check_template_input"]:
            if "template_mask_cb" not in feats or "template_force" not in feats:
                raise ValueError(
                    "TemplateReferencePotential requires template features but they are missing."
                )

            template_mask = feats["template_mask_cb"][feats["template_force"]]
            if template_mask.shape[0] == 0:
                raise ValueError(
                    f"TemplateReferencePotential requires template features but they are missing. Got template_mask shape {template_mask.shape}."
                )

        if "template_mask_cb" not in feats or "template_force" not in feats:
            return torch.empty([1, 0]), None, None, None, None

        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty([1, 0]), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                torch.arange(
                    feats["atom_pad_mask"].shape[1],
                    device=feats["atom_pad_mask"].device,
                    dtype=torch.float32,
                )[None, :, None],
            )
            .squeeze(-1)
            .long()
        )[0]
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )[0]

        index = torch.arange(
            template_mask.shape[-1], dtype=torch.long, device=template_mask.device
        )[None]
        upper_bounds = torch.full(
            template_mask.shape, float("inf"), device=index.device, dtype=torch.float32
        )
        ref_idxs = torch.argwhere(template_mask).T
        upper_bounds[ref_idxs.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][ref_idxs[0]]

        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class StructureReferencePotential(FlatBottomPotential, ReferencePotential):
    """
    RMSD-based potential that penalizes deviations from a final reference structure.
    
    This potential:
    1. Aligns the current structure to the reference using weighted rigid alignment
    2. Computes per-atom distances after alignment
    3. Applies a flat-bottom potential to penalize atoms beyond a threshold
    
    Required features in `feats`:
        - 'final_structure_coords': All-atom reference coordinates [n_atoms, 3]
        - 'valid_final_structure': Boolean indicating if reference is valid [1]
        
    Parameters:
        - 'threshold': Distance threshold for flat-bottom potential (Angstroms)
        - 'check_input': If True, raise error when reference features missing
        - 'use_rep_atoms_only': If True, only use representative atoms (e.g., CA) for alignment and penalty
    """
    
    def compute_args(self, feats, parameters):
        check_input = self.parameters.get('check_input', False)
        
        # Validate input
        if check_input:
            if 'final_structure_coords' not in feats or 'valid_final_structure' not in feats:
                raise ValueError(
                    "FinalStructureReferencePotential requires 'final_structure_coords' and "
                    "'valid_final_structure' features but they are missing."
                )
            if not feats['valid_final_structure']:
                raise ValueError(
                    "FinalStructureReferencePotential requires valid final structure but "
                    "'valid_final_structure' is False."
                )
        
        # Early exit if features missing or invalid
        if 'final_structure_coords' not in feats or 'valid_final_structure' not in feats:
            return torch.empty([1, 0], device=feats['atom_pad_mask'].device), None, None, None, None
        
        if not feats['valid_final_structure']:
            return torch.empty([1, 0], device=feats['atom_pad_mask'].device), None, None, None, None
        
        # Get reference coordinates [n_atoms, 3] -> [1, n_atoms, 3]
        ref_coords_all_atom = torch.zeros(
            1, feats['atom_pad_mask'].shape[1], 3, device=feats['atom_pad_mask'].device
        )
        ref_coords_all_atom[0, feats['atom_pad_mask'][0].bool()] = feats['final_structure_coords']   
     
        # Get atom padding mask [batch, n_atoms]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        
        use_rep_atoms_only = self.parameters.get('use_rep_atoms_only', True)
        
        if use_rep_atoms_only:
            # Use representative atoms (e.g., CA) for alignment
            # token_to_rep_atom: [batch, n_tokens, n_atoms] - one-hot mapping
            
            # Get representative atom index for each token
            ref_atom_index = (
                torch.bmm(
                    feats["token_to_rep_atom"].float(),
                    torch.arange(
                        feats["atom_pad_mask"].shape[1],
                        device=feats["atom_pad_mask"].device,
                        dtype=torch.float32,
                    )[None, :, None],
                )
                .squeeze(-1)
                .long()
            )[0]  # [n_tokens]
            
            # Extract representative atom coordinates from reference
            ref_coords = ref_coords_all_atom[:, ref_atom_index, :]  # [1, n_tokens, 3]
            
            # Mask: valid tokens (based on token padding)
            token_pad_mask = feats.get('token_pad_mask', None)
            if token_pad_mask is not None:
                ref_mask = token_pad_mask[0].bool()  # [n_tokens]
            else:
                # Infer from atom mask
                ref_mask = atom_pad_mask[ref_atom_index]  # [n_tokens]
            
            # Also check that reference coords are not all zeros
            ref_mask = ref_mask & (ref_coords[0].abs().sum(dim=-1) > 1e-6)
            ref_mask = ref_mask.unsqueeze(0)  # [1, n_tokens]
            
            # Token index for gradient propagation
            ref_token_index = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["token_index"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )[0]  # [n_atoms]
            
            n_tokens = ref_mask.shape[-1]
            index = torch.arange(n_tokens, dtype=torch.long, device=ref_mask.device)[None]
            
        else:
            # Use all atoms
            ref_coords = ref_coords_all_atom  # [1, n_atoms, 3]
            ref_mask = atom_pad_mask.unsqueeze(0)  # [1, n_atoms]
            
            # Also check that reference coords are not all zeros
            ref_mask = ref_mask & (ref_coords[0].abs().sum(dim=-1) > 1e-6)
            
            # For all-atom, indices are just atom indices
            ref_atom_index = torch.arange(
                atom_pad_mask.shape[0], 
                device=atom_pad_mask.device
            )  # [n_atoms]
            ref_token_index = None  # Not needed for all-atom
            
            n_atoms = ref_mask.shape[-1]
            index = torch.arange(n_atoms, dtype=torch.long, device=ref_mask.device)[None]
        
        # Check if we have any valid atoms
        if not ref_mask.any():
            return torch.empty([1, 0], device=ref_mask.device), None, None, None, None
        
        # Set up bounds
        lower_bounds = None
        threshold = parameters['threshold']
        upper_bounds = torch.full(
            ref_mask.shape, threshold, device=index.device, dtype=torch.float32
        )
        upper_bounds[~ref_mask] = float('inf')  # No penalty for masked positions
        
        k = torch.ones_like(upper_bounds)
        
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class GlobalStructureRMSDPotential(Potential):
    """
    Global RMSD potential that directly penalizes the RMSD value to a final reference structure.
    
    Unlike StructureReferencePotential which applies per-atom penalties,
    this potential computes a single RMSD value and applies a flat-bottom
    harmonic penalty to it.
    
    Required features in `feats`:
        - 'final_structure_coords': All-atom reference coordinates [n_atoms, 3]
        - 'valid_final_structure': Boolean indicating if reference is valid [1]
        
    Parameters:
        - 'threshold': RMSD threshold below which no penalty is applied (Angstroms)
        - 'k': Force constant for harmonic penalty
        - 'check_input': If True, raise error when reference features missing
        - 'use_rep_atoms_only': If True, only use representative atoms (e.g., CA)
    """
    
    def compute_args(self, feats, parameters):
        check_input = self.parameters.get('check_input', False)
        
        # Validate input
        if check_input:
            if 'final_structure_coords' not in feats or 'valid_final_structure' not in feats:
                raise ValueError(
                    "GlobalFinalStructureRMSDPotential requires 'final_structure_coords' and "
                    "'valid_final_structure' features but they are missing."
                )
            if not feats['valid_final_structure']:
                raise ValueError(
                    "GlobalFinalStructureRMSDPotential requires valid final structure but "
                    "'valid_final_structure' is False."
                )
        
        # Early exit if features missing or invalid
        if 'final_structure_coords' not in feats or 'valid_final_structure' not in feats:
            return torch.empty([1, 0], device=feats['atom_pad_mask'].device), None, None, None, None
        
        if not feats['valid_final_structure']:
            return torch.empty([1, 0], device=feats['atom_pad_mask'].device), None, None, None, None
        
        # Get reference coordinates [n_atoms, 3] -> [1, n_atoms, 3]
        ref_coords_all_atom = feats['final_structure_coords'].unsqueeze(0).clone()
        
        # Get atom padding mask [batch, n_atoms]
        atom_pad_mask = feats['atom_pad_mask'][0].bool()
        
        use_rep_atoms_only = self.parameters.get('use_rep_atoms_only', True)
        
        if use_rep_atoms_only:
            # Get representative atom index for each token
            ref_atom_index = (
                torch.bmm(
                    feats["token_to_rep_atom"].float(),
                    torch.arange(
                        feats["atom_pad_mask"].shape[1],
                        device=feats["atom_pad_mask"].device,
                        dtype=torch.float32,
                    )[None, :, None],
                )
                .squeeze(-1)
                .long()
            )[0]  # [n_tokens]
            
            # Extract representative atom coordinates from reference
            ref_coords = ref_coords_all_atom[:, ref_atom_index, :]  # [1, n_tokens, 3]
            
            # Mask: valid tokens
            token_pad_mask = feats.get('token_pad_mask', None)
            if token_pad_mask is not None:
                ref_mask = token_pad_mask[0].bool()
            else:
                ref_mask = atom_pad_mask[ref_atom_index]
            
            # Check reference coords are not all zeros
            ref_mask = ref_mask & (ref_coords[0].abs().sum(dim=-1) > 1e-6)
            ref_mask = ref_mask.unsqueeze(0)  # [1, n_tokens]
            
            # Token index for gradient propagation
            ref_token_index = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["token_index"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )[0]
            
            n_tokens = ref_mask.shape[-1]
            index = torch.arange(n_tokens, dtype=torch.long, device=ref_mask.device)[None]
            
        else:
            # Use all atoms
            ref_coords = ref_coords_all_atom
            ref_mask = atom_pad_mask.unsqueeze(0)
            ref_mask = ref_mask & (ref_coords[0].abs().sum(dim=-1) > 1e-6)
            
            ref_atom_index = torch.arange(
                atom_pad_mask.shape[0],
                device=atom_pad_mask.device
            )
            ref_token_index = None
            
            n_atoms = ref_mask.shape[-1]
            index = torch.arange(n_atoms, dtype=torch.long, device=ref_mask.device)[None]
        
        # Check if we have any valid atoms
        if not ref_mask.any():
            return torch.empty([1, 0], device=ref_mask.device), None, None, None, None
        
        k = parameters.get('k', 1.0)
        threshold = parameters.get('threshold', 0.0)
        
        return (
            index,
            (k, threshold),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )
    
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        """Compute RMSD after alignment."""
        # Align reference to current coords
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )
        
        # Compute squared differences
        diff = coords[:, index] - aligned_ref_coords  # [batch, n_points, 3]
        sq_diff = (diff ** 2).sum(dim=-1)  # [batch, n_points]
        
        # Masked mean for RMSD
        masked_sq_diff = sq_diff * ref_mask
        n_atoms = ref_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)  # [batch, 1]
        msd = masked_sq_diff.sum(dim=-1, keepdim=True) / n_atoms  # [batch, 1]
        rmsd = torch.sqrt(msd + 1e-8)  # [batch, 1]
        
        if not compute_gradient:
            return rmsd
        
        # Gradient of RMSD w.r.t. coords
        # d(RMSD)/d(x_i) = (x_i - x_i_ref) / (RMSD * N)
        # where N is the number of atoms
        grad_rmsd = diff * ref_mask.unsqueeze(-1) / (rmsd.unsqueeze(-1) * n_atoms.unsqueeze(-1) + 1e-8)
        grad = grad_rmsd.unsqueeze(1)  # [batch, 1, n_points, 3]
        
        return rmsd, grad
    
    def compute_function(
        self, value, k, threshold, negation_mask=None, compute_derivative=False
    ):
        """
        Flat-bottom harmonic penalty on RMSD.
        
        Energy = 0.5 * k * max(0, RMSD - threshold)^2
        """
        # value is RMSD: [batch, 1]
        deviation = torch.clamp(value - threshold, min=0)
        energy = 0.5 * k * deviation ** 2
        
        if not compute_derivative:
            return energy
        
        # dE/d(RMSD) = k * max(0, RMSD - threshold)
        dEnergy = k * deviation
        dEnergy[value <= threshold] = 0
        
        return energy, dEnergy


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = None
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )



def get_potentials(steering_args, boltz2=False):
    potentials = []
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if steering_args["physical_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.01
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if steering_args["contact_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1
                        if steering_args["contact_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "check_template_input": False,
                    }
                ),
            ]
        )
    return potentials
