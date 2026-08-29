"""Loading an exported model and the parameters it was exported with.

Every AID-MR program ships an ONNX file with a `<model>.json` sidecar holding
its preprocessing contract - input shape, frame count, windowing. The two are
loaded together on purpose: a session without its parameters is a model nobody
can feed correctly.

There were four copies of this. AMP's was the only one with session options, an
explicit provider list and a CPU fallback that re-raises rather than looping;
this is AMP's, with the unused imports dropped and a plural helper added for
AIFS, which loads two models at once.

`onnxruntime` vs `onnxruntime-gpu` is the caller's choice - depending on the
plain wheel here would fight the GPU wheel the FIRE container installs - so the
import is soft and the error names the extra.
"""

import json
from pathlib import Path

from loguru import logger

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    ort = None


def _require_ort():
    if ort is None:
        raise ImportError(
            "ONNX support needs onnxruntime: pip install 'aidmr-utils[onnx]', or "
            "install onnxruntime-gpu yourself if this is a CUDA host.")


#: Tried first, then CPU. CUDA is not fatal to miss: a developer machine and the
#: FIRE container should load the same model the same way.
DEFAULT_PROVIDERS = ["CUDAExecutionProvider"]


def load_model_onnx(model_path,
                    providers: list | None = None,
                    disable_graph_optimisation: bool = True):
    """Load an ONNX model and its JSON parameters.

    :param model_path: path to the model file; `<model_path>.json` must sit
        beside it.
    :param providers: ONNXRuntime providers for the session. `None` means
        `DEFAULT_PROVIDERS`, i.e. CUDA with a fallback to CPU.
    :param disable_graph_optimisation: disable ONNXRuntime graph optimisations.
        Defaults to True, which is what every AID-MR model has been exported and
        validated against. Passing `None` is an error rather than a silent
        default, so a caller threading a config value through cannot disable the
        optimisation setting by accident.
    :return: (session, params)
    """
    _require_ort()
    model_path = Path(model_path)

    if disable_graph_optimisation is None:
        raise ValueError("disable_graph_optimisation must be explicitly set")

    if providers is None:
        providers = list(DEFAULT_PROVIDERS)

    session_options = ort.SessionOptions()
    if disable_graph_optimisation:
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    try:
        session = ort.InferenceSession(str(model_path), session_options,
                                       providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        logger.warning(f"Failed to load ONNX model with providers {providers}, "
                       f"trying CPUExecutionProvider")
        session = ort.InferenceSession(str(model_path), session_options,
                                       providers=["CPUExecutionProvider"])

    with open(f"{model_path}.json", "r") as f:
        model_params = json.load(f)

    return session, model_params


def load_models_onnx(*model_paths, providers: list | None = None,
                     disable_graph_optimisation: bool = True):
    """Several models at once -> (sessions, params), as parallel tuples.

    AIFS runs a detector and a segmenter together and its `process()` signature
    takes them as a pair, so it wants this shape rather than a list of tuples.
    """
    loaded = [load_model_onnx(p, providers=providers,
                              disable_graph_optimisation=disable_graph_optimisation)
              for p in model_paths]
    sessions = tuple(s for s, _ in loaded)
    params = tuple(p for _, p in loaded)
    return sessions, params


def warm_up(session, sample_shape, name: str = 'model') -> None:
    """Run one dummy batch so the first real request does not pay the compile.

    Feeds the same batch to every input, which is what a multi-input model such
    as BPF's (x_2c, x_4c, identical shapes) wants and is a no-op distinction for
    a single-input one. Skipping multi-input models instead would push their
    first-call cost onto a live patient.
    """
    import numpy as np
    inputs = session.get_inputs()
    if not inputs:
        logger.warning(f"No inputs for {name} - skipping warm up")
        return
    batch = np.random.randn(1, *sample_shape).astype(np.float32)
    logger.debug(f"Warming up {name} with {batch.shape} x {len(inputs)} input(s)")
    session.run(None, {inp.name: batch for inp in inputs})
