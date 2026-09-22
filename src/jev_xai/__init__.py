"""jev-xai public API.

The public API is the names listed in ``__all__``.
"""

from __future__ import annotations

import logging

from jev_xai.adapters.callable import CallableAdapter
from jev_xai.adapters.capabilities import Diagnosis, diagnose
from jev_xai.adapters.jev import JEVAdapter
from jev_xai.config.loader import config_hash, load_config
from jev_xai.config.schema import JevXaiConfig
from jev_xai.errors import (
    BudgetExceededError,
    CapabilityError,
    JevXaiError,
    JevXaiUsageError,
    ModelCallError,
    ReplayMismatchError,
    SchemaVersionError,
)
from jev_xai.evidence.audit import build_audit_pack
from jev_xai.evidence.schema import SCHEMA_VERSION, DecisionRecord
from jev_xai.explainers.ablation import AblationExplainer
from jev_xai.explainers.context import ExplainContext, FeatureSpec
from jev_xai.explainers.counterfactual import CounterfactualExplainer
from jev_xai.limitations import LIMITATIONS
from jev_xai.model.client import ModelClient
from jev_xai.replay.diff import cross_version_diff
from jev_xai.replay.recorder import DecisionRecorder
from jev_xai.replay.repeatability import ReproducibilityProbe
from jev_xai.replay.replay import ReplayEngine
from jev_xai.stability.evaluator import StabilityEvaluator, assert_stable, explain_with_stability
from jev_xai.version import __version__

logger = logging.getLogger("jev_xai")
logger.addHandler(logging.NullHandler())

__all__ = [
    "SCHEMA_VERSION",
    "LIMITATIONS",
    "__version__",
    "AblationExplainer",
    "BudgetExceededError",
    "CallableAdapter",
    "CapabilityError",
    "CounterfactualExplainer",
    "DecisionRecord",
    "DecisionRecorder",
    "Diagnosis",
    "ExplainContext",
    "FeatureSpec",
    "JEVAdapter",
    "JevXaiConfig",
    "JevXaiError",
    "JevXaiUsageError",
    "ModelCallError",
    "ModelClient",
    "ReplayEngine",
    "ReplayMismatchError",
    "ReproducibilityProbe",
    "SchemaVersionError",
    "StabilityEvaluator",
    "assert_stable",
    "build_audit_pack",
    "config_hash",
    "cross_version_diff",
    "diagnose",
    "explain_with_stability",
    "load_config",
]
