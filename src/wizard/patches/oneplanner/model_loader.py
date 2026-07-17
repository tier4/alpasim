"""Model loading utilities for E2EPlanner.

Handles both fresh E2E checkpoints and legacy DiffusionPlanner checkpoints
(the latter need a ``pos_emb`` column migration from 14 → 15).
"""

import logging
import types

import torch

from oneplanner.deployment.checkpoint import extract_state_dict, normalize_state_dict

log = logging.getLogger(__name__)


def load_e2e_planner_weights(model, checkpoint_path: str, strict: bool = False):
    """Load weights into E2EPlanner from a checkpoint.

    Handles:
    - E2E checkpoint (direct load)
    - Legacy DiffusionPlanner checkpoint (pos_emb 14 → 15 column migration)

    Any state-dict entries for the removed ``perception_bridge`` module are
    silently discarded — older checkpoints may still carry them.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = extract_state_dict(ckpt)

    # Normalize DDP "module." AND Lightning "model." prefixes. The old code stripped
    # only "module.", so a Lightning E2E checkpoint (weights stored under "model.")
    # loaded with EVERY key still prefixed -> nothing matched -> the model was left
    # mostly randomly initialized. normalize_state_dict mirrors train_e2e's warm-start.
    sd = normalize_state_dict(sd)

    # Drop any dangling perception_bridge.* keys from legacy Mode 2 checkpoints.
    sd = {k: v for k, v in sd.items() if "perception_bridge" not in k}

    new_sd = {}
    for k, v in sd.items():
        # pos_emb migration: 14→15
        if "pos_emb.weight" in k and v.ndim == 2 and v.shape[1] == 14:
            padded = torch.zeros(v.shape[0], 15, dtype=v.dtype)
            padded[:, :14] = v
            v = padded
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)

    if missing:
        # WARNING, not info: missing keys mean layers at RANDOM INIT — in a
        # deployment context that must stay visible even when the caller never
        # configured logging (warnings reach stderr via logging's last-resort
        # handler; info is silently dropped). The print this replaced always
        # surfaced it (codex-connector on #82).
        log.warning("%d keys randomly initialised: %s", len(missing), missing[:5])
    if unexpected:
        log.warning("%d unexpected ckpt keys ignored: %s", len(unexpected), unexpected[:5])

    return model


def build_e2e_planner_from_checkpoint(checkpoint_path: str, config=None):
    """Build and load an E2EPlanner or DiffusionPlanner from a checkpoint.

    Constructs an E2EPlanner and loads the checkpoint on top.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    config : namespace, optional
        Model configuration.  When *None* a minimal default config is
        constructed from the checkpoint's ``config`` entry (if present)
        or from hard-coded defaults matching the standard architecture.

    Returns
    -------
    nn.Module
        The model with weights loaded, ready for training or inference.
    """
    from oneplanner.models.e2e import E2EPlanner
    from oneplanner.models.planner.normalizer import (
        ObservationNormalizer,
        StateNormalizer,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try to recover config from checkpoint
    if config is None:
        config = ckpt.get("config", None)

    # Build a minimal config if none was found
    if config is None:
        config = _default_config()

    # Ensure normalizers exist on the config.
    #
    # HOTFIX (alpasim): the shipped ``model_loader`` defaulted state_normalizer /
    # observation_normalizer to identity + empty, but the checkpoint was trained
    # with ``configs/planner/normalization.json`` (ego mean=[10,0,0,0],
    # std=[20,20,1,1] etc.). Identity normalizers at inference broke the input
    # scale seen by the DiT and made the decoder collapse to near-stationary
    # predictions regardless of inputs. Prefer the JSON built into the image;
    # fall back to identity only if the file is missing.
    _norm_json = getattr(config, "normalization_file_path", None)
    if not _norm_json:
        import os as _os
        _candidate = "/opt/oneplanner/configs/planner/normalization.json"
        if _os.path.exists(_candidate):
            _norm_json = _candidate
            config.normalization_file_path = _candidate

    if not hasattr(config, "state_normalizer") or config.state_normalizer is None:
        if _norm_json:
            log.info("Loading StateNormalizer from %s", _norm_json)
            config.state_normalizer = StateNormalizer.from_json(config)
        else:
            log.warning(
                "No normalization JSON found; falling back to identity "
                "StateNormalizer. Predictions may be miscalibrated."
            )
            config.state_normalizer = StateNormalizer(
                mean=[[[0.0, 0.0, 0.0, 0.0]]],
                std=[[[1.0, 1.0, 1.0, 1.0]]],
            )
    if not hasattr(config, "observation_normalizer") or config.observation_normalizer is None:
        if _norm_json:
            log.info("Loading ObservationNormalizer from %s", _norm_json)
            config.observation_normalizer = ObservationNormalizer.from_json(config)
        else:
            config.observation_normalizer = ObservationNormalizer({})

    model = E2EPlanner.from_config(config, bevfusion_head=None)

    # Load weights
    model = load_e2e_planner_weights(model, checkpoint_path, strict=False)

    return model


def _default_config():
    """Return a minimal config namespace for model construction.

    Values match the reference Diffusion-Planner argparse defaults in
    ``train_predictor.py``.
    """
    from oneplanner.constants import OUTPUT_T

    cfg = types.SimpleNamespace()
    cfg.hidden_dim = 256
    cfg.num_heads = 8
    cfg.query_dim = 128

    # Encoder config (matching train_predictor.py defaults)
    cfg.encoder_mixer_depth = 6
    cfg.encoder_fusion_depth = 6
    cfg.encoder_drop_path_rate = 0.1
    cfg.use_ego_history = True
    cfg.ego_history_dropout_rate = 0.6
    cfg.use_turn_indicators = True

    # Decoder config
    cfg.decoder_depth = 3
    cfg.decoder_drop_path_rate = 0.1
    cfg.future_len = OUTPUT_T
    cfg.diffusion_model_type = "x_start"
    cfg.use_velocity_representation = False
    cfg.guidance_fn = None
    cfg.guidance_scale = 0.5
    cfg.state_normalizer = None
    cfg.observation_normalizer = None

    # Normalization file path (default relative path, same as train_predictor.py)
    cfg.normalization_file_path = None

    # Deployment loads INFERENCE models: the latent world model is a
    # training-only auxiliary and grid-sample BEV needs a live BEV feature
    # map wired by the runtime. Pin both OFF so the train-side ON-defaults
    # (E2EConfig, 2026-06-11) never leak into checkpoint-only loading —
    # fill_defaults would otherwise set use_latent_wm=True and from_config
    # would raise on the missing latent_action_dim (Codex on #92).
    cfg.use_latent_wm = False
    cfg.latent_action_dim = None

    return cfg
