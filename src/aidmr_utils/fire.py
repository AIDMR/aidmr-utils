"""The FIRE connection: replaying it, and the save-and-return passthrough.

The seam every AID-MR program sits behind. AIDMR-FIRE's server hands each
configured program the same connection in turn, so the connection has to be
replayable; and `save_and_return` is the no-model config used to capture what a
scanner actually sent.

THE PROCESS CONTRACT
--------------------
A program's `process()` is called as

    process(connection, config, metadata, log_dir=..., ort_session=...,
            model_params=...)

and returns either `images` or `(images, aidmr_dict)`. When it returns the pair,
the server rehydrates the dict with `aidmr_utils.result.AIDMRResult.from_dict`
and writes it to the DB - which is why the writer's and the host's aidmr-utils
must be compatible, and why `from_dict` now checks `utilsVersion` rather than
trusting that six vendored copies stayed byte-identical.

Needs the `mrd` extra.
"""

import traceback
from types import NoneType

from loguru import logger

try:
    import ismrmrd
except ImportError:  # pragma: no cover - exercised by the extras, not the suite
    ismrmrd = None


class ReplayableConnection:
    """Wrap a FIRE Connection to allow multiple passes over incoming data.

    The first iteration reads from the underlying connection while caching
    the messages. Subsequent iterations replay the cached messages so that
    multiple configs can process the same input images.

    A PROGRAM THAT STOPS EARLY TRUNCATES THE REPLAY FOR EVERY PROGRAM AFTER IT.
    The caching iterator marks itself complete in a `finally`, so breaking out of
    the loop - which drops the generator and closes it - leaves `_cached` True
    with a partial buffer. The next config in `amp__cmrq__bpf` then sees only the
    messages the first one happened to consume, with nothing to say so.

    That is the behaviour as it has always been, and `tests/test_phase5.py` pins
    it rather than changing it: which programs read to exhaustion is a property
    of those programs, and quietly making the buffer complete here would change
    what every one of them is fed. If a program ever does need to stop early, the
    fix is for it to drain the iterator first, not for this to paper over it.
    """

    def __init__(self, connection):
        self._connection = connection
        self._buffer = []
        self._cached = False

    def __iter__(self):
        if self._cached:
            # Return a new iterator over the cached messages
            return iter(self._buffer)
        return self._caching_iter()

    def _caching_iter(self):
        try:
            for item in self._connection:
                self._buffer.append(item)
                yield item
        finally:
            # Mark data as cached once the iterator is consumed
            self._cached = True

    def send_image(self, *args, **kwargs):
        return self._connection.send_image(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._connection, attr)


def save_and_return(connection, *args, **kwargs):
    incoming_mrd_images = []
    try:
        for item in connection:
            if not isinstance(item, (ismrmrd.Image, ismrmrd.Acquisition, ismrmrd.Waveform, NoneType)):
                logger.error(f"Unsupported data type {type(item).__name__}")
                break

            if item is None:
                # Sequence finished; trigger processing
                break

            if isinstance(item, ismrmrd.Image):
                incoming_mrd_images.append(item)
            else:
                continue

        logger.info(f"Received {len(incoming_mrd_images)} images from scanner")

        for i_image, img in enumerate(incoming_mrd_images):
            meta = ismrmrd.Meta.deserialize(img.attribute_string)
            logger.info(f"Image {i_image}: {img.image_type=} {img.data.shape=} {meta['SequenceDescription']}")
            logger.info(f"{meta=}")

        connection.send_image(incoming_mrd_images)

    except Exception as e:
        logger.error(traceback.format_exc())
        raise ValueError(f"This is bad: {e} {traceback.format_exc()}")

    finally:
        connection.send_close()
