import multiprocessing
import os
import pickle
import platform
import tarfile
import time
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm
import json

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from boltz.model.models.boltz2 import Boltz2
from boltz.model.wrappers import WalkJumpWrapper, DiffusionWrapper, NoisedSeedWrapper, TransitionPathWrapper
from dotenv import find_dotenv, load_dotenv

# Machine-specific paths (datasets, MSAs, results) are referenced as ${VAR} in
# input configs and resolved from a .env file; variables already set in the
# shell take precedence.
load_dotenv(find_dotenv(usecwd=True), override=False)

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20
    structure_guidance: bool = False
    structure_guidance_weight: float = 0.1

@dataclass
class WalkJumpArgs:
    """Walk-jump parameters."""

    steps: int
    sigma: float
    delta: float
    friction: float
    save_every_n_steps: int
    num_denoising_steps: int = 1
    diversity_mode: str = "none"
    diversity_height: float = 1.0
    diversity_width: float = 2.0
    diversity_deposit_every: int = 20
    diversity_bias_factor: float = 0.0
    M: float = 1.0
    save_trajectory: bool = True
    burn_in_steps: int = 0
    verbose: bool = False
    cpu_offload: bool = False
    v_init: str = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: Optional[float] = None


@rank_zero_only
def download_boltz1(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        click.echo(
            f"Downloading the CCD data to {tar_mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
    if not mols.exists():
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the Boltz-2 weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    # Check if existing predictions are found (only top-level prediction folders)
    pred_dir = outdir / "predictions"
    if pred_dir.exists():
        existing = {d.name for d in pred_dir.iterdir() if d.is_dir()}
    else:
        existing = set()

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        msg = (
            f"Found some existing predictions ({len(existing)}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = f"Found {len(existing)} existing predictions, will override."
        click.echo(msg)

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    click.echo("Checking input data for affinity.")

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (outdir / "predictions" / r.id / f"affinity_{r.id}.json").exists()
    }

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "affinity predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing affinity predictions, will override."
        click.echo(msg)

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")

    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {"Content-Type": "application/json", key: value}
        click.echo(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        click.echo("Using basic authentication for MSA server")
    else:
        click.echo("No authentication provided for MSA server")

    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    try:
        # Parse data
        if path.suffix.lower() in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix.lower() in (".yml", ".yaml"):
            target = parse_yaml(path, ccd, mol_dir, boltz2)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)  # noqa: TRY301
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)  # noqa: TRY301

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None

    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    # if records_dir.exists():
    #     # Load existing records
    #     existing = [Record.load(p) for p in records_dir.glob("*.json")]
    #     processed_ids = {record.id for record in existing}

    #     # Filter to missing only
    #     data = [d for d in data if d.stem not in processed_ids]

    #     # Nothing to do, update the manifest and return
    #     if data:
    #         click.echo(
    #             f"Found {len(existing)} existing processed inputs, skipping them."
    #         )
    #     else:
    #         click.echo("All inputs are already processed.")
    #         updated_manifest = Manifest(existing)
    #         updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    records = [Record.load(p) for p in records_dir.glob("*.json")]
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help=" to dump the s and z embeddings into a npz file. Default is False.",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    write_embeddings: bool = False,
) -> None:
    """Run predictions with Boltz."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")

        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_subsample_msa={subsample_msa}_num_subsampled={num_subsampled_msa}_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = (
            "fork"
            if platform.system() != "win32" and platform.system() != "Windows"
            else "spawn"
        )
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=model == "boltz2",
        write_embeddings=write_embeddings,
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        click.echo(msg)

        # Create data module
        if model == "boltz2":
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        else:
            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
            )

        # Load model
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        # Compute structure predictions
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            return

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False

        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
            affinity_mw_correction=affinity_mw_correction,
        )
        model_module.eval()

        trainer.callbacks[0] = pred_writer
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )


@click.group()
def boltz_jump_cli():
    """Boltz-Jump CLI."""
    pass


@boltz_jump_cli.command("predict")
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz2"]),
    help="The model to use for prediction. Only boltz2 is supported here.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is 'md'.",
    default="md",
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help="Whether to dump the s and z embeddings into a npz file. Default is False.",
)
@click.option(
    "--walk_jump_sigma",
    type=float,
    default=2.0,
    help="Noise level (in Angstroms) for walk trajectory.",
)
@click.option(
    "--walk_jump_steps", type=int, default=100, help="Number of walk-jump steps."
)
@click.option(
    "--walk_jump_delta",
    type=float,
    default=1.0,
    help="Step size for the BAOAB integrator.",
)
@click.option(
    "--walk_jump_friction",
    type=float,
    default=0.1,
    help="Friction coefficient for the BAOAB integrator.",
)
@click.option(
    "--walk_jump_save_every_n_steps",
    type=int,
    default=1,
    help="Frequency of saving frames in the walk-jump trajectory.",
)
@click.option(
    "--num_denoising_steps",
    type=int,
    default=1,
    help="Number of denoising steps to perform after the walk-jump sampling. Default is 1 (i.e. one final denoising step).",
)
@click.option(
    "--diversity_mode",
    type=click.Choice(["none", "seed", "metad", "q_ratchet"]),
    default="none",
    help="Bias for the walk: repel from the seed, metadynamics in RMSD, or a ratchet on the fraction of "
    "native contacts of final_structure (see wrappers/diversity.py).",
)
@click.option("--diversity_height", type=float, default=1.0,
              help="Hill height in kT per atom (seed, metad); ratchet stiffness k in kT per residue (q_ratchet).")
@click.option("--diversity_width", type=float, default=2.0, help="Hill width, as RMSD in Angstrom.")
@click.option("--diversity_deposit_every", type=int, default=20, help="Walk steps between hills (metad).")
@click.option("--diversity_bias_factor", type=float, default=0.0,
              help="Well-tempered bias factor (>1 enables tempering; metad).")
@click.option(
    "--use_full_diffusion",
    is_flag=True,
    help="Whether to use the full diffusion, instead of walk-jump."
)
@click.option(
    "--num_diffusion_samples", type=int, default=100, help="Number of diffusion samples to generate per input when using full diffusion."
)
@click.option(
    "--use_noised_seed",
    is_flag=True,
    help="Shortened reverse diffusion baseline: noise seed to sigma, then denoise. Mutually exclusive with --use_full_diffusion.",
)
@click.option(
    "--num_noised_seed_samples",
    type=int,
    default=100,
    help="Number of independent noised seeds to generate and denoise. Used with --use_noised_seed.",
)
@click.option(
    "--noised_seed_ode",
    is_flag=True,
    help="Use deterministic ODE (DDIM-style) instead of stochastic SDE. Used with --use_noised_seed.",
)
@click.option(
    "--noised_seed_one_step",
    is_flag=True,
    help="Single Tweedie denoising step at sigma (walk without the jump). Used with --use_noised_seed.",
)
@click.option(
    "--structure_guidance",
    is_flag=True,
    help="Whether to use structure guidance for steering. Default is False.",
)
@click.option(
    "--structure_guidance_weight",
    type=float,
    default=0.01,
    help="Weight for structure guidance when steering. Default is 0.1.",
)
def predict_walk_jump(
    data: str,
    out_dir: str,
    cache: str,
    checkpoint: Optional[str],
    devices: int,
    accelerator: str,
    recycling_steps: int,
    sampling_steps: int,
    step_scale: Optional[float],
    write_full_pae: bool,
    write_full_pde: bool,
    output_format: Literal["pdb", "mmcif"],
    num_workers: int,
    override: bool,
    seed: Optional[int],
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    use_potentials: bool,
    model: str,
    method: Optional[str],
    preprocessing_threads: int,
    max_msa_seqs: int,
    subsample_msa: bool,
    num_subsampled_msa: int,
    no_kernels: bool,
    write_embeddings: bool,
    walk_jump_sigma: float,
    walk_jump_steps: int,
    walk_jump_delta: float,
    walk_jump_friction: float,
    walk_jump_save_every_n_steps: int,
    num_denoising_steps: int,
    diversity_mode: str = "none",
    diversity_height: float = 1.0,
    diversity_width: float = 2.0,
    diversity_deposit_every: int = 20,
    diversity_bias_factor: float = 0.0,
    use_full_diffusion: bool = False,
    num_diffusion_samples: Optional[int] = None,
    use_noised_seed: bool = False,
    num_noised_seed_samples: int = 100,

    noised_seed_ode: bool = False,
    noised_seed_one_step: bool = False,
    structure_guidance: bool = False,
    structure_guidance_weight: float = 0.1,
) -> None:
    """Runs walk-jump sampling with Boltz-2."""
    if accelerator == "cpu":
        click.echo("Running on CPU, this will be slow. Consider using a GPU.")

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")

        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")
    click.echo(f"Using checkpoint: {checkpoint}")

    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    if use_full_diffusion:
        out_dir = out_dir / f"boltz_diffusion_results_{data.stem}"
    elif use_noised_seed:
        if noised_seed_one_step:
            mode_tag = "one_step"
        elif noised_seed_ode:
            mode_tag = "ode"
        else:
            mode_tag = "sde"
        out_dir = out_dir / f"boltz_noised_seed_{mode_tag}_s={walk_jump_sigma}_n={num_noised_seed_samples}_{data.stem}"
    else:
        # Walk-denoise (multi-step jump) runs get their own directory; single-step
        # runs keep the original name.
        denoise_tag = f"_n={num_denoising_steps}" if num_denoising_steps > 1 else ""
        if diversity_mode == "seed":
            denoise_tag += f"_div=seed_h={diversity_height}_w={diversity_width}"
        elif diversity_mode == "q_ratchet":
            denoise_tag += f"_div=qratchet_k={diversity_height}"
        elif diversity_mode == "metad":
            denoise_tag += (f"_div=metad_h={diversity_height}_w={diversity_width}"
                            f"_k={diversity_deposit_every}_bf={diversity_bias_factor}")
        out_dir = out_dir / f"boltz_jump_results_s={walk_jump_sigma}_d={walk_jump_delta}_f={walk_jump_friction}{denoise_tag}_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = check_inputs(data)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=True,
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=out_dir, override=override
    )
    if not filtered_manifest.records:
        click.echo("All predictions already exist. Use --override to re-run.")
        return

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=method,
    )

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params = Boltz2DiffusionParams(step_scale=step_scale)
    pairformer_args = PairformerArgsV2()
    steering_args = BoltzSteeringParams(
        fk_steering=False,
        physical_guidance_update=use_potentials,
        contact_guidance_update=use_potentials,
        structure_guidance=structure_guidance,
        structure_guidance_weight=structure_guidance_weight,
    )
    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )
    walk_jump_args = WalkJumpArgs(
        steps=walk_jump_steps,
        sigma=walk_jump_sigma,
        delta=walk_jump_delta,
        friction=walk_jump_friction,
        save_every_n_steps=walk_jump_save_every_n_steps,
        num_denoising_steps=num_denoising_steps,
        diversity_mode=diversity_mode,
        diversity_height=diversity_height,
        diversity_width=diversity_width,
        diversity_deposit_every=diversity_deposit_every,
        diversity_bias_factor=diversity_bias_factor,
    )

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        map_location="cpu",
        ema=False,
        use_kernels=not no_kernels,
        predict_args=predict_args,
        diffusion_process_args=asdict(diffusion_params),
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()

    # Move model to device; we do this explicitly since we don't use Lightning's trainer.
    device = torch.device(
        "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Create wrapper for walk-jump sampling.
    if use_full_diffusion:
        wrapper = DiffusionWrapper(
            model=model,
            predict_args=predict_args,
            steering_args=asdict(steering_args),
            num_samples=num_diffusion_samples,
            use_kernels=not no_kernels,
        )
    elif use_noised_seed:
        wrapper = NoisedSeedWrapper(
            model=model,
            predict_args=predict_args,
            steering_args=asdict(steering_args),
            noised_seed_args={
                "sigma": walk_jump_sigma,
                "num_samples": num_noised_seed_samples,
                "deterministic": noised_seed_ode,
                "one_step": noised_seed_one_step,
            },
            use_kernels=not no_kernels,
        )
    else:
        wrapper = WalkJumpWrapper(
            model=model,
            predict_args=predict_args,
            steering_args=asdict(steering_args),
            walk_jump_args=asdict(walk_jump_args),
            use_kernels=not no_kernels,
        )

    predict_loader = data_module.predict_dataloader()
    for feats in predict_loader:
        start_time = time.perf_counter()
        record_id = feats["record"][0].id
        click.echo(f"Processing: {record_id}")

        if use_full_diffusion:
            click.echo(
                f"Starting full diffusion sampling for {num_diffusion_samples} samples..."
            )
        elif use_noised_seed:
            if noised_seed_one_step:
                click.echo(f"Starting noised-seed one-step sampling: {num_noised_seed_samples} samples at sigma={walk_jump_sigma} Å...")
            else:
                mode = "ODE" if noised_seed_ode else "SDE"
                click.echo(f"Starting noised-seed {mode} sampling: {num_noised_seed_samples} samples from sigma={walk_jump_sigma} Å...")
        else:
            click.echo(f"Starting walk-jump sampling for {walk_jump_steps} steps...")

        result = wrapper.process_batch(feats)
        # DiffusionWrapper returns (x_init, xhat_traj); WalkJumpWrapper returns (x_init, xhat_traj, y_traj)
        x_init, xhat_traj = result[0], result[1]
        y_traj = result[2] if len(result) > 2 else None
        click.echo("Sampling complete.")

        pred_out_dir = out_dir / "predictions"
        writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=pred_out_dir,
            output_format=output_format,
            boltz2=True,
        )

        # Write the initial sample (x_init)
        writer.write_on_batch_end(
            prediction={
                "masks": feats["atom_pad_mask"],
                "coords": x_init,
                "exception": False,
            },
            batch=feats,
            trainer=None,
            pl_module=None,
            batch_indices=None,
            batch_idx=None,
            dataloader_idx=None,
        )

        output_format_extension = "cif" if output_format == "mmcif" else "pdb"
        output_topology_file = (
            pred_out_dir / record_id / f"{record_id}_model_0.{output_format_extension}"
        )
        if not output_topology_file.exists():
            click.echo(
                f"Error: Initial output file was not written to {output_topology_file}",
                err=True,
            )
            continue

        # Write xhat trajectory (denoised samples).
        output_trajectory_file = pred_out_dir / record_id / f"{record_id}_model_0.xtc"
        wrapper.write_trajectory(
            traj=xhat_traj,
            feats=feats,
            topology_file=output_topology_file,
            trajectory_file=output_trajectory_file,
        )

        # Write noisy y trajectory only for walk-jump (not noised-seed or full diffusion).
        if y_traj is not None:
            output_y_trajectory_file = pred_out_dir / record_id / f"{record_id}_model_0_y.xtc"
            wrapper.write_trajectory(
                traj=y_traj,
                feats=feats,
                topology_file=output_topology_file,
                trajectory_file=output_y_trajectory_file,
            )

        click.echo(f"Predictions for {record_id} written to disk:")
        click.echo(f"- Topology: {output_topology_file.absolute()}")
        click.echo(f"- Trajectory: {output_trajectory_file.absolute()}")
        if y_traj is not None:
            click.echo(f"- Y Trajectory: {output_y_trajectory_file.absolute()}")
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        click.echo(f"Elapsed time for {record_id}: {elapsed:.2f} seconds\n")


@click.group()
def boltz_path_cli():
    """Boltz-Transition-Path CLI."""
    pass


@boltz_path_cli.command("predict")
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    default="./",
    help="Output directory for predictions.",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    default=get_cache_path,
    help="Cache directory for model/data.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="Optional checkpoint path.",
)
@click.option("--accelerator", type=click.Choice(["gpu", "cpu"]), default="gpu")
@click.option("--recycling_steps", type=int, default=3)
@click.option("--sampling_steps", type=int, default=200)
@click.option("--step_scale", type=float, default=None)
@click.option("--output_format", type=click.Choice(["pdb", "mmcif"]), default="mmcif")
@click.option("--num_workers", type=int, default=2)
@click.option("--override", is_flag=True)
@click.option("--seed", type=int, default=None)
@click.option("--use_msa_server", is_flag=True)
@click.option("--msa_server_url", type=str, default="https://api.colabfold.com")
@click.option("--msa_pairing_strategy", type=str, default="greedy")
@click.option("--preprocessing-threads", type=int, default=multiprocessing.cpu_count())
@click.option("--max_msa_seqs", type=int, default=8192)
@click.option("--subsample_msa", is_flag=True)
@click.option("--num_subsampled_msa", type=int, default=1024)
@click.option("--no_kernels", is_flag=True)
# OM-TPS options
@click.option("--om_sigma", type=float, default=0.5, help="Noise level for score evaluation.")
@click.option("--om_path_length", type=int, default=100, help="Number of path points (L).")
@click.option("--om_delta_t", type=float, default=0.1, help="Timestep (Δt).")
@click.option("--om_friction", type=float, default=0.1, help="Friction (ζ).")
@click.option("--om_diffusivity", type=float, default=1.0, help="Diffusivity (D).")
@click.option("--om_optimization_steps", type=int, default=100, help="Optimization steps.")
@click.option("--om_learning_rate", type=float, default=0.05, help="Learning rate.")
@click.option("--om_action_type", type=click.Choice(["truncated", "full"]), default="truncated")
@click.option("--om_use_divergence", is_flag=True, help="Include divergence term.")
@click.option("--om_hutchinson_samples", type=int, default=1)
@click.option("--om_num_paths", type=int, default=1, help="Number of paths to sample.")
def predict_transition_path(
    data: str,
    out_dir: str,
    cache: str,
    checkpoint: Optional[str],
    accelerator: str,
    recycling_steps: int,
    sampling_steps: int,
    step_scale: Optional[float],
    output_format: Literal["pdb", "mmcif"],
    num_workers: int,
    override: bool,
    seed: Optional[int],
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    preprocessing_threads: int,
    max_msa_seqs: int,
    subsample_msa: bool,
    num_subsampled_msa: int,
    no_kernels: bool,
    # OM-TPS specific
    om_sigma: float,
    om_path_length: int,
    om_delta_t: float,
    om_friction: float,
    om_diffusivity: float,
    om_optimization_steps: int,
    om_learning_rate: float,
    om_action_type: str,
    om_use_divergence: bool,
    om_hutchinson_samples: int,
    om_num_paths: int,
) -> None:
    """
    Run transition path sampling between two conformations using OM action minimization.
    
    DATA should contain both initial (coords) and final (final_coords) structures.
    """
    if accelerator == "cpu":
        click.echo("Running on CPU, this will be slow.")

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if seed is not None:
        seed_everything(seed)

    # Setup cache
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"
    click.echo(f"Using checkpoint: {checkpoint}")

    # Setup paths
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_tps_L={om_path_length}_s={om_sigma}_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process inputs
    data = check_inputs(data)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        boltz2=True,
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=out_dir, override=override)
    if not filtered_manifest.records:
        click.echo("All predictions exist. Use --override to re-run.")
        return

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
        template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
    )

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    # Model config
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    step_scale = 1.5 if step_scale is None else step_scale
    diffusion_params = Boltz2DiffusionParams(step_scale=step_scale)
    pairformer_args = PairformerArgsV2()
    steering_args = BoltzSteeringParams()
    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=True,
    )

    om_tps_args = dict(
        num_paths=om_num_paths,
        sigma=om_sigma,
        path_length=om_path_length,
        delta_t=om_delta_t,
        friction=om_friction,
        diffusivity=om_diffusivity,
        optimization_steps=om_optimization_steps,
        learning_rate=om_learning_rate,
        action_type=om_action_type,
        use_divergence=om_use_divergence,
        hutchinson_samples=om_hutchinson_samples,
    )

    # Load model
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        map_location="cpu",
        ema=False,
        use_kernels=not no_kernels,
        predict_args=predict_args,
        diffusion_process_args=asdict(diffusion_params),
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()

    device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    model.to(device)

    wrapper = TransitionPathWrapper(
        model=model,
        predict_args=predict_args,
        steering_args=asdict(steering_args),
        om_tps_args=om_tps_args,
        use_kernels=not no_kernels,
    )

    predict_loader = data_module.predict_dataloader()
    for feats in predict_loader:
        start_time = time.perf_counter()
        record_id = feats["record"][0].id
        click.echo(f"\nProcessing: {record_id}")
        click.echo(f"  Path length: {om_path_length}, Sigma: {om_sigma}, Steps: {om_optimization_steps}")

        all_paths = []
        all_histories = []
        
        for path_idx in range(om_num_paths):
            if om_num_paths > 1:
                click.echo(f"\nPath {path_idx + 1}/{om_num_paths}:")
                if seed is not None:
                    seed_everything(seed + path_idx)
            
            path, history = wrapper.process_batch(feats)
            all_paths.append(path)
            all_histories.append(history)

        click.echo("\nOptimization complete.")

        # Write outputs
        pred_out_dir = out_dir / "predictions"
        writer = BoltzWriter(
            data_dir=processed.targets_dir,
            output_dir=pred_out_dir,
            output_format=output_format,
            boltz2=True,
        )

        # Write starting and final structures for topology
        feats_repeated = feats.copy()
        feats_repeated["atom_pad_mask"] = feats["atom_pad_mask"].unsqueeze(0).expand(2, -1, -1)
        feats_repeated["record"] = feats["record"] * 2
        writer.write_on_batch_end(
            prediction={
                "masks": feats_repeated["atom_pad_mask"],
                "coords": torch.stack([path[0], path[-1]]),
                "exception": False,
            },
            batch=feats_repeated,
            trainer=None,
            pl_module=None,
            batch_indices=None,
            batch_idx=None,
            dataloader_idx=None,
        )
        ext = "cif" if output_format == "mmcif" else "pdb"
        topology_file = pred_out_dir / record_id / f"{record_id}_model_0.{ext}"
        final_topology_file = pred_out_dir / record_id / f"{record_id}_model_1.{ext}"

        # Write trajectories
        for path_idx, path in enumerate(all_paths):
            traj_file = pred_out_dir / record_id / f"{record_id}_path_{path_idx}_final.xtc"
            wrapper.write_trajectory(
                xhat_traj=path,
                feats=feats,
                topology_file=topology_file,
                trajectory_file=traj_file,
            )
            click.echo(f"  Path {path_idx}:")
            click.echo(f"    Starting topology: {topology_file.absolute()}")
            click.echo(f"    Final topology: {final_topology_file.absolute()}")
            click.echo(f"    Trajectory: {traj_file.absolute()}")

        # Write out intermediate paths in history
        for path_idx, history in enumerate(all_histories):
            history_steps = [entry['step'] for entry in history]
            history_paths = [entry['path'] for entry in history]
            for step, path in zip(history_steps, history_paths):
                traj_file = pred_out_dir / record_id / f"{record_id}_path_{path_idx}_step_{step}.xtc"
                wrapper.write_trajectory(
                    xhat_traj=path,
                    feats=feats,
                    topology_file=topology_file,
                    trajectory_file=traj_file,
                )
    
        # Save history after removing path data
        for history in all_histories:
            for entry in history:
                entry.pop('path', None)
        history_file = pred_out_dir / record_id / f"{record_id}_history.json"
        with open(history_file, "w") as f:
            json.dump({"config": om_tps_args, "paths": all_histories}, f, indent=2)

        elapsed = time.perf_counter() - start_time
        click.echo(f"History: {history_file}")
        click.echo(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    cli()
