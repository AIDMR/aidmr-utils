"""The phase-5 modules: onnx, imaging, fire."""

import json

import numpy as np
import pytest

from aidmr_utils import fire as fire_mod
from aidmr_utils import onnx as onnx_mod
from aidmr_utils.fire import ReplayableConnection
from aidmr_utils.onnx import load_model_onnx, load_models_onnx, warm_up


# ---------------------------------------------------------------------------
# onnx
# ---------------------------------------------------------------------------

class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Enough of an InferenceSession to exercise warm_up."""

    def __init__(self, input_names=('input',)):
        self._inputs = [FakeInput(n) for n in input_names]
        self.fed = None

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        self.fed = feed
        return []


def test_none_graph_optimisation_is_an_error(tmp_path):
    """A config value threaded through must not silently pick a default."""
    with pytest.raises(ValueError, match='explicitly set'):
        load_model_onnx(tmp_path / 'm.onnx', disable_graph_optimisation=None)


def test_helpful_error_when_onnxruntime_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(onnx_mod, 'ort', None)
    with pytest.raises(ImportError, match='onnxruntime'):
        load_model_onnx(tmp_path / 'm.onnx')


def test_params_are_loaded_from_the_json_sidecar(monkeypatch, tmp_path):
    """A session without its parameters is a model nobody can feed correctly."""
    model = tmp_path / 'm.onnx'
    model.write_bytes(b'')
    (tmp_path / 'm.onnx.json').write_text(json.dumps({'windowing': '0_100',
                                                      'n_frames': 24}))

    class FakeOrt:
        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = 'disabled'

        @staticmethod
        def SessionOptions():
            class O:
                graph_optimization_level = None
            return O()

        @staticmethod
        def InferenceSession(path, options, providers):
            return FakeSession()

    monkeypatch.setattr(onnx_mod, 'ort', FakeOrt)
    session, params = load_model_onnx(model)
    assert params == {'windowing': '0_100', 'n_frames': 24}
    assert isinstance(session, FakeSession)


def test_cpu_fallback_is_tried_then_gives_up(monkeypatch, tmp_path):
    model = tmp_path / 'm.onnx'
    model.write_bytes(b'')
    (tmp_path / 'm.onnx.json').write_text('{}')
    attempts = []

    class FakeOrt:
        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = 'disabled'

        @staticmethod
        def SessionOptions():
            class O:
                graph_optimization_level = None
            return O()

        @staticmethod
        def InferenceSession(path, options, providers):
            attempts.append(providers)
            if providers == ['CUDAExecutionProvider']:
                raise RuntimeError('no CUDA here')
            return FakeSession()

    monkeypatch.setattr(onnx_mod, 'ort', FakeOrt)
    load_model_onnx(model)
    assert attempts == [['CUDAExecutionProvider'], ['CPUExecutionProvider']]


def test_a_cpu_only_failure_is_raised_not_looped(monkeypatch, tmp_path):
    model = tmp_path / 'm.onnx'
    model.write_bytes(b'')

    class FakeOrt:
        class GraphOptimizationLevel:
            ORT_DISABLE_ALL = 'disabled'

        @staticmethod
        def SessionOptions():
            class O:
                graph_optimization_level = None
            return O()

        @staticmethod
        def InferenceSession(path, options, providers):
            raise RuntimeError('broken model')

    monkeypatch.setattr(onnx_mod, 'ort', FakeOrt)
    with pytest.raises(RuntimeError, match='broken model'):
        load_model_onnx(model, providers=['CPUExecutionProvider'])


def test_load_models_onnx_returns_parallel_tuples(monkeypatch, tmp_path):
    """AIFS's process() takes (sessions, params) as a pair, not a list of pairs."""
    calls = []

    def fake_single(path, providers=None, disable_graph_optimisation=True):
        calls.append(path)
        return FakeSession(), {'which': str(path)}

    monkeypatch.setattr(onnx_mod, 'load_model_onnx', fake_single)
    sessions, params = load_models_onnx('yolo.onnx', 'seg.onnx')
    assert len(sessions) == 2 and len(params) == 2
    assert params[0]['which'] == 'yolo.onnx' and params[1]['which'] == 'seg.onnx'


def test_warm_up_feeds_every_input():
    """BPF's session takes two inputs of identical shape; both must be fed."""
    s = FakeSession(input_names=('x_2c', 'x_4c'))
    warm_up(s, (1, 24, 192, 192), name='bpf')
    assert set(s.fed) == {'x_2c', 'x_4c'}
    assert s.fed['x_2c'].shape == (1, 1, 24, 192, 192)
    assert s.fed['x_2c'].dtype == np.float32


def test_warm_up_on_a_session_with_no_inputs_is_a_warning_not_a_crash():
    s = FakeSession(input_names=())
    warm_up(s, (1, 8, 8))
    assert s.fed is None


# ---------------------------------------------------------------------------
# imaging
# ---------------------------------------------------------------------------

imaging = pytest.importorskip('aidmr_utils.imaging',
                              reason="needs the [imaging] extra")


def test_put_text_round_trips_a_float_image():
    """A float image in [0, 1] must come back in [0, 1], not 0-255."""
    out = imaging.put_text_on_img(np.zeros((64, 64), np.float32), 'AMP 4ch')
    assert out.max() <= 1.0
    assert out.max() > 0, "nothing was drawn"


def test_put_text_leaves_a_uint8_image_as_uint8():
    out = imaging.put_text_on_img(np.zeros((64, 64), np.uint8) + 10, 'x')
    assert out.max() > 10


@pytest.mark.parametrize('fn,args', [
    ('add_cross_to_array', ((16, 16),)),
    ('add_dashed_cross_to_array', ((16, 16),)),
])
def test_markers_draw_something(fn, args):
    a = np.zeros((32, 32), np.uint8)
    getattr(imaging, fn)(a, *args, val=255)
    assert (a > 0).sum() > 0


def test_lines_draw_something():
    a = np.zeros((32, 32), np.uint8)
    imaging.add_line_to_array(a, (2, 2), (28, 28), val=255)
    assert (a > 0).sum() > 0
    b = np.zeros((32, 32), np.uint8)
    imaging.add_dotted_line_to_array(b, (2, 2), (28, 28), val=255)
    assert 0 < (b > 0).sum()


def test_a_marker_outside_the_array_does_not_raise():
    """Keypoints can land off a cropped preview."""
    a = np.zeros((16, 16), np.uint8)
    imaging.add_cross_to_array(a, (100, 100), val=255)
    imaging.add_line_to_array(a, (-50, -50), (100, 100), val=255)


# ---------------------------------------------------------------------------
# fire
# ---------------------------------------------------------------------------

class FakeConnection:
    def __init__(self, items):
        self.items = items
        self.reads = 0
        self.sent = []

    def __iter__(self):
        self.reads += 1
        return iter(self.items)

    def send_image(self, images):
        self.sent.append(images)

    def send_close(self):
        self.sent.append('close')


def test_replay_reads_the_connection_once():
    """The server hands the same connection to each configured program in turn."""
    c = FakeConnection(['a', 'b', 'c'])
    rc = ReplayableConnection(c)
    assert list(rc) == ['a', 'b', 'c']
    assert list(rc) == ['a', 'b', 'c']
    assert list(rc) == ['a', 'b', 'c']
    assert c.reads == 1, "the underlying connection was re-read"


def test_replay_forwards_unknown_attributes():
    c = FakeConnection([])
    c.savedata = True
    assert ReplayableConnection(c).savedata is True


def test_replay_forwards_send_image():
    c = FakeConnection([])
    ReplayableConnection(c).send_image(['img'])
    assert c.sent == [['img']]


def test_stopping_early_truncates_the_replay_for_everyone_after():
    """Pins a real hazard rather than asserting desirable behaviour.

    The caching iterator marks itself complete in a `finally`, so a program that
    breaks out of the loop leaves a partial buffer that every later config in
    `amp__cmrq__bpf` then replays. See the note on ReplayableConnection: this is
    how it has always behaved, and changing it here would change what every
    program is fed.
    """
    c = FakeConnection(['a', 'b', 'c'])
    rc = ReplayableConnection(c)
    for item in rc:
        if item == 'b':
            break
    # the generator was abandoned; whatever it cached is what replays
    assert list(rc) == ['a', 'b']
    assert c.reads == 1
