# EVS + AMR/AMR-WB Codec Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime RTP decode support for `AMR-NB/8000`, `AMR-WB/16000`, and `EVS/16000` (EVS Primary and EVS AMR-WB IO), including all codec modes, Compact/Header-Full EVS payloads, SID/DTX/NO_DATA, AMR CRC/robust-sorting/interleaving, and a real EVS decoder backend.

**Architecture:** Keep the existing `CodecSpec -> depacketizer -> EncodedFrame -> CodecDecoder -> PCM` pipeline. Add EVS to the codec registry and SDP parser, add an EVS depacketizer that emits bit-accurate `EncodedFrame`s, and add a decoder backend that converts EVS frames to PCM. Because stock FFmpeg/PyAV has no EVS decoder, use the 3GPP reference `EVS_dec` executable through a G.192 bitstream adapter. Keep AMR/AMR-WB decoding on the existing FFmpeg decoders and complete the missing AMR payload features.

**Tech Stack:** Python 3.10, PyAV/FFmpeg, NumPy, FastAPI, pytest, Docker, 3GPP EVS reference decoder (`EVS_dec`).

**Spec:** `docs/superpowers/specs/2026-10-21-bad-audio-detector-design.md` and the RTP support matrix in `README.md` section 6.

## How to Execute This Plan

Work top to bottom. Do not start a task until the previous task's tests pass.
Each task has its own failing test, implementation, passing test, and commit.
If a command fails, fix the failure before moving on. Use `python3`, not
`python`. The repository root is `/Users/ogi/Projects/bad_audio_detector`.

## Global Constraints

- Target detector sample rate is `16000` Hz.- EVS RTP clock is always `16000` Hz, even when decoded output is NB/WB/SWB/FB.
- AMR-NB RTP clock is `8000`; AMR-WB RTP clock is `16000`.
- One AMR/AMR-WB/EVS speech frame is `20 ms`.
- At the 16 kHz target rate, one 20 ms frame is `320` samples.
- AMR-NB frame duration is `160` samples at 8 kHz; AMR-WB/EVS frame duration is `320` samples at 16 kHz.
- Do not silently return empty PCM when a codec decoder is unavailable. Raise a clear error.
- Do not add new PyPI dependencies for this work.
- Preserve Python 3.10 compatibility and existing style (`from __future__ import annotations`, type hints).
- Run tests with `python3 -m pytest`; do not rely on a `python` executable.
- Existing tests in `tests/unit/test_depacketizers.py` that use `depacketize(packet, spec)` must keep working.
- EVS encoding for synthetic training data is out of scope. Runtime decoding is in scope.

---

## File Structure

**Create**
- `src/bad_audio/io/rtp/depacketizer/evs.py` — EVS Compact/Header-Full depacketizer and bit helpers.
- `tests/unit/test_evs_specs.py` — EVS registry and SDP tests.
- `tests/unit/test_evs_depacketizer.py` — EVS payload parsing tests.
- `tests/unit/test_evs_decoder.py` — EVS decoder backend tests.
- `tests/fixtures/fake_evs_dec.py` — deterministic fake `EVS_dec` executable for tests.

**Modify**
- `src/bad_audio/config.py` — EVS decoder settings.
- `src/bad_audio/io/rtp/specs.py` — EVS alias/default and new `CodecSpec` fields; EVS SDP value helpers.
- `src/bad_audio/io/rtp/sdp.py` — EVS/AMR SDP/fmtp parsing.
- `src/bad_audio/io/rtp/depacketizer/base.py` — `flush()` and bit metadata on `EncodedFrame`.
- `src/bad_audio/io/rtp/depacketizer/__init__.py` — EVS registration and per-session depacketizer instances.
- `src/bad_audio/io/rtp/depacketizer/amr.py` — mode-set enforcement, robust sorting, interleaving.
- `src/bad_audio/io/rtp/decoder.py` — EVS decoder backend, G.192 adapter, clear errors.
- `src/bad_audio/pipeline.py` — pass EVS settings to `CodecDecoder`.
- `src/bad_audio/streaming/session.py` — own a depacketizer instance, EVS config passthrough, flush on finish.
- `tests/unit/test_config.py` — EVS settings defaults.
- `tests/unit/test_depacketizers.py` — AMR mode-set/robust-sorting/interleaving tests.
- `tests/integration/test_api.py` — end-to-end EVS RTP WebSocket test.
- `Dockerfile` — build/install `EVS_dec`.
- `.env.example` — EVS decoder settings.
- `README.md` — support matrix, SDP examples, troubleshooting.

---

## Task 1: Add EVS settings and codec registry entry

**Files:**
- Modify: `src/bad_audio/config.py:8-41`
- Modify: `src/bad_audio/io/rtp/specs.py:6-53`
- Modify: `.env.example`
- Modify: `tests/unit/test_config.py`
- Test: `tests/unit/test_evs_specs.py`

**Interfaces:**
- Produces:
  - `Settings.evs_decoder_cmd: str`
  - `Settings.evs_decoder_rate: int`
  - `Settings.evs_decoder_timeout_s: float`
  - `CodecSpec.hf_only: bool`
  - `CodecSpec.evs_mode: str`
  - `CodecSpec.evs_bitrates: tuple[float, ...] | None`
  - `CodecSpec.evs_bandwidths: tuple[str, ...] | None`
  - `CodecSpec.evs_dtx: bool | None`
  - `CodecSpec.evs_cmr: int | None`
  - `CodecSpec.evs_channel_aware: int | None`
  - `CodecSpec.evs_max_red: int | None`
  - `CodecSpec.interleaving_max: int`
  - `normalize_encoding("EVS") == "EVS"`
  - `make_spec("EVS")` returns `CodecSpec(encoding="EVS", rtp_clock_rate=16000, sample_rate=16000, decoder="evs")`

- [ ] **Step 1: Write the failing settings test**

Add to `tests/unit/test_config.py`:

```python
def test_evs_decoder_defaults():
    from bad_audio.config import Settings

    settings = Settings(_env_file=None)
    assert settings.evs_decoder_cmd == "EVS_dec"
    assert settings.evs_decoder_rate == 16000
    assert settings.evs_decoder_timeout_s == 5.0
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_config.py::test_evs_decoder_defaults -v
```

Expected: `AttributeError: 'Settings' object has no attribute 'evs_decoder_cmd'`.

- [ ] **Step 3: Add settings fields**

In `src/bad_audio/config.py`, add after `payload_type_map`:

```python
    evs_decoder_cmd: str = "EVS_dec"
    evs_decoder_rate: int = 16000
    evs_decoder_timeout_s: float = 5.0
```

Add to `.env.example` after the RTP settings:

```env
# EVS decoder backend (3GPP reference decoder executable)
EVS_DECODER_CMD=EVS_dec
EVS_DECODER_RATE=16000
EVS_DECODER_TIMEOUT_S=5.0
```

- [ ] **Step 4: Write the failing registry test**

Create `tests/unit/test_evs_specs.py`:

```python
from __future__ import annotations

from bad_audio.io.rtp.specs import make_spec, normalize_encoding


def test_evs_alias_and_defaults():
    assert normalize_encoding("evs") == "EVS"
    assert normalize_encoding("EVS-WB") == "EVS"

    spec = make_spec("EVS")
    assert spec.encoding == "EVS"
    assert spec.rtp_clock_rate == 16000
    assert spec.sample_rate == 16000
    assert spec.channels == 1
    assert spec.decoder == "evs"
    assert spec.hf_only is False
    assert spec.evs_mode == "auto"
```

- [ ] **Step 5: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_specs.py::test_evs_alias_and_defaults -v
```

Expected: `ValueError: unsupported codec 'evs'`.

- [ ] **Step 6: Add `CodecSpec` fields and EVS registry/default**

In `src/bad_audio/io/rtp/specs.py`, extend `CodecSpec`:

```python
@dataclass(frozen=True, slots=True)
class CodecSpec:
    encoding: str
    rtp_clock_rate: int
    sample_rate: int
    channels: int = 1
    octet_align: bool = False
    amr_crc: bool = False
    interleaving: bool = False
    interleaving_max: int = 0
    robust_sorting: bool = False
    mode_set: tuple[int, ...] | None = None
    decoder: str | None = None
    max_frames_per_packet: int = 10
    hf_only: bool = False
    evs_mode: str = "auto"
    evs_bitrates: tuple[float, ...] | None = None
    evs_bandwidths: tuple[str, ...] | None = None
    evs_dtx: bool | None = None
    evs_cmr: int | None = None
    evs_channel_aware: int | None = None
    evs_max_red: int | None = None
```

Add aliases in `_ALIASES`:

```python
    "EVS": "EVS",
    "EVS-WB": "EVS",
    "EVS-NB": "EVS",
    "EVS-SWB": "EVS",
    "EVS-FB": "EVS",
```

Add to `_DEFAULTS`:

```python
    "EVS": CodecSpec("EVS", 16000, 16000, decoder="evs", evs_mode="auto"),
```

- [ ] **Step 7: Add EVS SDP value helpers**

In `src/bad_audio/io/rtp/specs.py`, add:

```python
def parse_evs_bitrates(value: str) -> tuple[float, ...]:
    parts = value.strip().split("-")
    if len(parts) == 1:
        return (float(parts[0]) * 1000.0,)
    if len(parts) == 2:
        return (float(parts[0]) * 1000.0, float(parts[1]) * 1000.0)
    raise ValueError(f"invalid EVS bit-rate range {value!r}")


def parse_evs_bandwidths(value: str) -> tuple[str, ...]:
    key = value.strip().lower()
    ranges = {
        "nb-fb": ("nb", "wb", "swb", "fb"),
        "nb-swb": ("nb", "wb", "swb"),
        "nb-wb": ("nb", "wb"),
    }
    if key in ranges:
        return ranges[key]
    if key in ("nb", "wb", "swb", "fb"):
        return (key,)
    raise ValueError(f"invalid EVS bandwidth {value!r}")
```

- [ ] **Step 8: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_config.py tests/unit/test_evs_specs.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/bad_audio/config.py src/bad_audio/io/rtp/specs.py .env.example tests/unit/test_config.py tests/unit/test_evs_specs.py
git commit -m "feat: add EVS codec registry and decoder settings"
```

---

## Task 2: Parse EVS SDP and fmtp parameters

**Files:**
- Modify: `src/bad_audio/io/rtp/sdp.py:39-64`
- Modify: `src/bad_audio/io/rtp/specs.py` (helpers already added in Task 1)
- Test: `tests/unit/test_evs_specs.py`

**Interfaces:**
- Consumes: `CodecSpec` fields from Task 1; `parse_evs_bitrates()` and `parse_evs_bandwidths()` from Task 1.
- Produces: `parse_sdp(text)` returns an `EVS` `CodecSpec` with `hf_only`, `evs_mode`, `evs_bitrates`, `evs_bandwidths`, `evs_dtx`, `evs_cmr`, `evs_channel_aware`, `evs_max_red`, and `interleaving_max` populated.

- [ ] **Step 1: Write the failing SDP test**

Add to `tests/unit/test_evs_specs.py`. First update the module imports at the top of the file to:

```python
from __future__ import annotations

import pytest

from bad_audio.io.rtp.sdp import parse_sdp
from bad_audio.io.rtp.specs import make_spec, normalize_encoding
```

Then add:

```python
def test_parse_evs_sdp_parameters():
    sdp = """v=0
m=audio 49170 RTP/AVP 96
a=rtpmap:96 EVS/16000/1
a=fmtp:96 evs-mode-switch=1; hf-only=1; dtx=0; br=5.9-24.4; bw=nb-wb; cmr=1; ch-aw-recv=2; max-red=220
"""
    specs = parse_sdp(sdp)
    spec = specs[96]

    assert spec.encoding == "EVS"
    assert spec.rtp_clock_rate == 16000
    assert spec.sample_rate == 16000
    assert spec.channels == 1
    assert spec.hf_only is True
    assert spec.evs_mode == "amr-wb-io"
    assert spec.evs_bitrates == (5900.0, 24400.0)
    assert spec.evs_bandwidths == ("nb", "wb")
    assert spec.evs_dtx is False
    assert spec.evs_cmr == 1
    assert spec.evs_channel_aware == 2
    assert spec.evs_max_red == 220
```

Add an AMR validation test:

```python
def test_amr_sdp_rejects_wrong_clock_rate():
    sdp = """v=0
m=audio 49170 RTP/AVP 97
a=rtpmap:97 AMR-NB/16000/1
"""
    with pytest.raises(ValueError, match="AMR-NB.*8000"):
        parse_sdp(sdp)
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_specs.py -v
```

Expected: the EVS test fails because `evs_mode` is still `auto`; the AMR test fails because no clock validation exists.

- [ ] **Step 3: Add EVS and AMR validation to SDP parsing**

In `src/bad_audio/io/rtp/sdp.py`, import the helpers:

```python
from .specs import (
    CodecSpec,
    make_spec,
    normalize_encoding,
    parse_evs_bandwidths,
    parse_evs_bitrates,
    spec_from_pt,
)
```

Replace the block after `canonical = normalize_encoding(name)` with:

```python
        if canonical == "AMR-NB" and clock != 8000:
            raise ValueError("AMR-NB RTP clock rate must be 8000")
        if canonical == "AMR-WB" and clock != 16000:
            raise ValueError("AMR-WB RTP clock rate must be 16000")
        if canonical == "OPUS":
            overrides["sample_rate"] = 48000
        if canonical in ("AMR-NB", "AMR-WB"):
            overrides["octet_align"] = params.get("octet-align", "0") == "1"
            overrides["amr_crc"] = params.get("crc", "0") == "1"
            overrides["robust_sorting"] = params.get("robust-sorting", "0") == "1"
            interleaving = int(params.get("interleaving", "0") or "0")
            overrides["interleaving"] = interleaving > 0
            overrides["interleaving_max"] = interleaving
            mode_set = params.get("mode-set")
            if mode_set:
                overrides["mode_set"] = tuple(int(v) for v in mode_set.split(",") if v)
        if canonical == "EVS":
            if clock != 16000:
                raise ValueError("EVS RTP clock rate must be 16000")
            overrides["rtp_clock_rate"] = 16000
            overrides["sample_rate"] = 16000
            overrides["hf_only"] = params.get("hf-only", "0") == "1"
            overrides["evs_mode"] = (
                "amr-wb-io" if params.get("evs-mode-switch", "0") == "1" else "primary"
            )
            if "dtx" in params:
                overrides["evs_dtx"] = params["dtx"] == "1"
            if "cmr" in params:
                overrides["evs_cmr"] = int(params["cmr"])
            if "ch-aw-recv" in params:
                overrides["evs_channel_aware"] = int(params["ch-aw-recv"])
            if "max-red" in params:
                overrides["evs_max_red"] = int(params["max-red"])
            bitrate_value = (
                params.get("br") or params.get("br-recv") or params.get("br-send")
            )
            if bitrate_value:
                overrides["evs_bitrates"] = parse_evs_bitrates(bitrate_value)
            bandwidth_value = (
                params.get("bw") or params.get("bw-recv") or params.get("bw-send")
            )
            if bandwidth_value:
                overrides["evs_bandwidths"] = parse_evs_bandwidths(bandwidth_value)
            if "ch-recv" in params or "ch-send" in params:
                overrides["channels"] = max(
                    int(params.get("ch-send", "1")),
                    int(params.get("ch-recv", "1")),
                )
            mode_set = params.get("mode-set")
            if mode_set:
                overrides["mode_set"] = tuple(int(v) for v in mode_set.split(",") if v)
```

- [ ] **Step 4: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_evs_specs.py tests/unit/test_depacketizers.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bad_audio/io/rtp/sdp.py tests/unit/test_evs_specs.py
git commit -m "feat: parse EVS SDP parameters"
```

---

## Task 3: Add EVS Compact-format depacketizer

**Files:**
- Modify: `src/bad_audio/io/rtp/depacketizer/base.py`
- Create: `src/bad_audio/io/rtp/depacketizer/evs.py`
- Modify: `src/bad_audio/io/rtp/depacketizer/__init__.py:15-34`
- Test: `tests/unit/test_evs_depacketizer.py`

**Interfaces:**
- Consumes: `EncodedFrame`, `CodecSpec`, `RtpPacket`.
- Produces:
  - `EncodedFrame.bit_offset: int`
  - `EncodedFrame.bit_length: int`
  - `EncodedFrame.mode: str`
  - `EvsDepacketizer` with `depacketize(packet, spec) -> list[EncodedFrame]`
  - `get_depacketizer(spec)` returns `EvsDepacketizer` for `encoding == "EVS"`.

- [ ] **Step 1: Write the failing Compact-format tests**

Create `tests/unit/test_evs_depacketizer.py`:

```python
from __future__ import annotations

import pytest

from bad_audio.io.rtp.depacketizer import depacketize, make_spec
from bad_audio.io.rtp.depacketizer.evs import (
    AMRWB_IO_BITS,
    COMPACT_SIZES,
    PRIMARY_BITS,
)
from bad_audio.io.rtp.header import RtpPacket


def rtp(payload: bytes, *, pt: int = 96, ts: int = 1000) -> RtpPacket:
    return RtpPacket(
        version=2,
        padding=False,
        extension=False,
        csrc_count=0,
        marker=False,
        payload_type=pt,
        sequence=1,
        timestamp=ts,
        ssrc=1,
        payload=payload,
        payload_offset=12,
    )


def pack_bits(fields: list[tuple[int, int]]) -> bytes:
    value = 0
    nbits = 0
    for bits, width in fields:
        value = (value << width) | (bits & ((1 << width) - 1))
        nbits += width
    pad = (-nbits) % 8
    value <<= pad
    nbits += pad
    return value.to_bytes(nbits // 8, "big")


def test_evs_compact_primary_8k():
    payload = bytes(range(20))
    frames = depacketize(rtp(payload), make_spec("EVS"))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.codec == "EVS"
    assert frame.data == payload
    assert frame.bit_offset == 0
    assert frame.bit_length == 160
    assert frame.duration == 320
    assert frame.sample_rate == 16000
    assert frame.frame_type == "speech"
    assert frame.mode == "primary"
    assert frame.rtp_timestamp == 1000


def test_evs_compact_primary_sid():
    frames = depacketize(rtp(b"\x00" * 6), make_spec("EVS"))

    assert len(frames) == 1
    assert frames[0].bit_length == 48
    assert frames[0].frame_type == "sid"
    assert frames[0].speech is False


def test_evs_compact_primary_28k_56_bit_is_not_sid():
    frames = depacketize(rtp(b"\x00" * 7), make_spec("EVS"))

    assert len(frames) == 1
    assert frames[0].bit_length == 56
    assert frames[0].frame_type == "speech"
    assert frames[0].speech is True


@pytest.mark.parametrize("total_bits", sorted(COMPACT_SIZES))
def test_evs_compact_all_protected_sizes(total_bits: int):
    mode, ft_index = COMPACT_SIZES[total_bits]
    payload = bytes((total_bits + 7) // 8)
    frames = depacketize(rtp(payload), make_spec("EVS"))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.duration == 320
    assert frame.sample_rate == 16000
    if mode == "primary":
        assert frame.bit_length == PRIMARY_BITS[ft_index]
        assert frame.mode == "primary"
    else:
        assert frame.bit_length == AMRWB_IO_BITS[ft_index]
        assert frame.mode == "amr-wb-io"


def test_evs_compact_amr_wb_io_rotates_d0_to_front():
    ft_index = 2
    k = AMRWB_IO_BITS[ft_index]
    d0 = 1

    payload = pack_bits([(0, 3)] + [(1, 1)] + [(0, k - 2)] + [(d0, 1)])
    frames = depacketize(rtp(payload), make_spec("EVS"))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.bit_length == k
    assert frame.mode == "amr-wb-io"
    assert (frame.data[0] >> 7) & 1 == d0
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_depacketizer.py -v
```

Expected: import error for `bad_audio.io.rtp.depacketizer.evs`.

- [ ] **Step 3: Add bit metadata to `EncodedFrame`**

In `src/bad_audio/io/rtp/depacketizer/base.py`:

```python
@dataclass(slots=True)
class EncodedFrame:
    codec: str
    data: bytes
    rtp_timestamp: int
    duration: int
    sample_rate: int
    channels: int = 1
    valid: bool = True
    speech: bool = True
    frame_type: str = "speech"
    sequence: int = 0
    bit_offset: int = 0
    bit_length: int = 0
    mode: str = ""
```

- [ ] **Step 4: Implement `evs.py` with Compact parsing**

Create `src/bad_audio/io/rtp/depacketizer/evs.py`:

```python
from __future__ import annotations

from .base import Depacketizer, EncodedFrame

EVS_CLOCK_RATE = 16000
EVS_FRAME_SAMPLES = 320

# TS 26.445 Table A.4: EVS Primary frame types.
PRIMARY_BITS: dict[int, int | None] = {
    0: 56, 1: 144, 2: 160, 3: 192, 4: 264, 5: 328, 6: 488, 7: 640,
    8: 960, 9: 1280, 10: 1920, 11: 2560, 12: 48, 13: None, 14: 0, 15: 0,
}

# TS 26.445 Table A.5: EVS AMR-WB IO frame types.
AMRWB_IO_BITS: dict[int, int | None] = {
    0: 132, 1: 177, 2: 253, 3: 285, 4: 317, 5: 365, 6: 397, 7: 461,
    8: 477, 9: 40, 10: None, 11: None, 12: None, 13: None, 14: 0, 15: 0,
}

# TS 26.445 Table A.1: protected Compact-format payload sizes in bits.
# The second item is the frame-type index used by the relevant bit table.
COMPACT_SIZES: dict[int, tuple[str, int]] = {
    48: ("primary", 12),
    56: ("primary", 0),
    136: ("amr-wb-io", 0),
    144: ("primary", 1),
    160: ("primary", 2),
    184: ("amr-wb-io", 1),
    192: ("primary", 3),
    256: ("amr-wb-io", 2),
    264: ("primary", 4),
    288: ("amr-wb-io", 3),
    320: ("amr-wb-io", 4),
    328: ("primary", 5),
    368: ("amr-wb-io", 5),
    400: ("amr-wb-io", 6),
    464: ("amr-wb-io", 7),
    480: ("amr-wb-io", 8),
    488: ("primary", 6),
    640: ("primary", 7),
    960: ("primary", 8),
    1280: ("primary", 9),
    1920: ("primary", 10),
    2560: ("primary", 11),
}


def read_bits(data: bytes, offset: int, length: int) -> list[int]:
    if offset < 0 or length < 0 or offset + length > len(data) * 8:
        raise ValueError("truncated EVS payload")
    return [
        (data[(offset + index) >> 3] >> (7 - ((offset + index) & 7))) & 1
        for index in range(length)
    ]


def pack_bits(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            out[index >> 3] |= 1 << (7 - (index & 7))
    return bytes(out)


def rotate_last_bit_to_front(data: bytes, offset: int, length: int) -> tuple[bytes, int]:
    bits = read_bits(data, offset, length)
    return pack_bits(bits[-1:] + bits[:-1]), length


class EvsDepacketizer(Depacketizer):
    def depacketize(self, packet, spec) -> list[EncodedFrame]:
        payload = packet.payload
        if not payload:
            return []
        if spec.channels != 1:
            raise NotImplementedError("multichannel EVS depacketization is not supported")

        total_bits = len(payload) * 8
        if not spec.hf_only:
            if total_bits == 56:
                if not (payload[0] & 0x80):
                    return self._parse_compact(packet, spec, total_bits)
                return self._parse_header_full(packet, spec)
            if total_bits in COMPACT_SIZES:
                return self._parse_compact(packet, spec, total_bits)
        return self._parse_header_full(packet, spec)

    def _parse_compact(self, packet, spec, total_bits: int) -> list[EncodedFrame]:
        mode, ft_index = COMPACT_SIZES[total_bits]
        if mode == "primary":
            bit_length = PRIMARY_BITS[ft_index]
            if bit_length is None:
                raise ValueError(f"unsupported EVS primary frame type {ft_index}")
            return [
                EncodedFrame(
                    codec="EVS",
                    data=packet.payload,
                    rtp_timestamp=packet.timestamp,
                    duration=EVS_FRAME_SAMPLES,
                    sample_rate=EVS_CLOCK_RATE,
                    valid=True,
                    speech=bit_length != 48,
                    frame_type="sid" if bit_length == 48 else "speech",
                    sequence=packet.sequence,
                    bit_offset=0,
                    bit_length=bit_length,
                    mode="primary",
                )
            ]

        frame_bits = AMRWB_IO_BITS[ft_index]
        if frame_bits is None:
            raise ValueError(f"unsupported EVS AMR-WB IO frame type {ft_index}")
        data, bit_length = rotate_last_bit_to_front(packet.payload, 3, frame_bits)
        return [
            EncodedFrame(
                codec="EVS",
                data=data,
                rtp_timestamp=packet.timestamp,
                duration=EVS_FRAME_SAMPLES,
                sample_rate=EVS_CLOCK_RATE,
                valid=True,
                speech=True,
                frame_type="speech",
                sequence=packet.sequence,
                bit_offset=0,
                bit_length=bit_length,
                mode="amr-wb-io",
            )
        ]

    def _parse_header_full(self, packet, spec) -> list[EncodedFrame]:
        raise NotImplementedError("EVS Header-Full format is implemented in Task 4")
```

- [ ] **Step 5: Register EVS in the depacketizer factory**

In `src/bad_audio/io/rtp/depacketizer/__init__.py`, add the import and branch:

```python
from .evs import EvsDepacketizer
```

```python
    if encoding == "EVS":
        return EvsDepacketizer()
```

- [ ] **Step 6: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_evs_depacketizer.py -v
```

Expected: PASS for all Compact tests. The Header-Full tests do not exist yet.

- [ ] **Step 7: Commit**

```bash
git add src/bad_audio/io/rtp/depacketizer/base.py src/bad_audio/io/rtp/depacketizer/evs.py src/bad_audio/io/rtp/depacketizer/__init__.py tests/unit/test_evs_depacketizer.py
git commit -m "feat: add EVS compact-format depacketizer"
```

---

## Task 4: Add EVS Header-Full format and AMR-WB IO SID parsing

**Files:**
- Modify: `src/bad_audio/io/rtp/depacketizer/evs.py`
- Test: `tests/unit/test_evs_depacketizer.py`

**Interfaces:**
- Consumes: `PRIMARY_BITS`, `AMRWB_IO_BITS`, `read_bits()`, `EncodedFrame`.
- Produces: `EvsDepacketizer._parse_header_full(packet, spec)` handles CMR bytes, one or more ToC bytes, EVS Primary and EVS AMR-WB IO frames, SID, SPEECH_LOST, NO_DATA, and 20 ms timestamp stepping.

- [ ] **Step 1: Write the failing Header-Full tests**

Add to `tests/unit/test_evs_depacketizer.py`:

```python
def test_evs_header_full_single_primary_frame():
    payload = bytes([0x02]) + bytes(range(20))
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.data == bytes(range(20))
    assert frame.bit_length == 160
    assert frame.frame_type == "speech"
    assert frame.mode == "primary"


def test_evs_header_full_multiframe():
    payload = bytes([0x42, 0x04]) + b"A" * 20 + b"B" * 33
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 2
    assert [frame.rtp_timestamp for frame in frames] == [1000, 1320]
    assert [frame.bit_length for frame in frames] == [160, 264]
    assert frames[0].data == b"A" * 20
    assert frames[1].data == b"B" * 33


def test_evs_header_full_with_cmr_byte():
    payload = bytes([0x80, 0x02]) + bytes(20)
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 1
    assert frames[0].bit_length == 160


def test_evs_header_full_amr_wb_io_sid_56_bit():
    payload = bytes([0x80, 0x29]) + bytes([0x11]) * 5
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.mode == "amr-wb-io"
    assert frame.frame_type == "sid"
    assert frame.speech is False
    assert frame.bit_length == 40


def test_evs_header_full_no_data():
    frames = depacketize(
        rtp(bytes([0x0F])),
        make_spec("EVS", hf_only=True),
    )

    assert len(frames) == 1
    assert frames[0].valid is False
    assert frames[0].frame_type == "no_data"
    assert frames[0].bit_length == 0


@pytest.mark.parametrize("ft_index,bits", sorted(PRIMARY_BITS.items()))
def test_evs_header_full_all_primary_frame_types(ft_index: int, bits: int | None):
    if bits is None:
        pytest.skip("reserved EVS primary frame type")
    payload = bytes([ft_index]) + bytes((bits + 7) // 8)
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 1
    assert frames[0].mode == "primary"
    assert frames[0].bit_length == bits


@pytest.mark.parametrize("ft_index,bits", sorted(AMRWB_IO_BITS.items()))
def test_evs_header_full_all_amr_wb_io_frame_types(ft_index: int, bits: int | None):
    if bits is None:
        pytest.skip("reserved EVS AMR-WB IO frame type")
    payload = bytes([0x80, 0x20 | ft_index]) + bytes((bits + 7) // 8)
    frames = depacketize(rtp(payload), make_spec("EVS", hf_only=True))

    assert len(frames) == 1
    assert frames[0].mode == "amr-wb-io"
    assert frames[0].bit_length == bits
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_depacketizer.py -v
```

Expected: `NotImplementedError: EVS Header-Full format is implemented in Task 4`.

- [ ] **Step 3: Implement `_parse_header_full`**

Replace the stub in `src/bad_audio/io/rtp/depacketizer/evs.py` with:

```python
    def _parse_header_full(self, packet, spec) -> list[EncodedFrame]:
        payload = packet.payload
        if not payload:
            return []

        offset_bits = 0
        if payload[0] & 0x80:
            offset_bits = 8

        toc: list[int] = []
        while True:
            if offset_bits + 8 > len(payload) * 8:
                raise ValueError("truncated EVS ToC")
            header = read_bits(payload, offset_bits, 8)[0]
            offset_bits += 8
            if header & 0x80:
                raise ValueError("unexpected CMR byte in EVS ToC")
            follow = (header >> 6) & 1
            toc.append(header & 0x3F)
            if len(toc) > spec.max_frames_per_packet:
                raise ValueError("too many EVS frames in one packet")
            if not follow:
                break

        frames: list[EncodedFrame] = []
        for index, ft in enumerate(toc):
            mode_bit = (ft >> 5) & 1
            ft_index = ft & 0x0F
            bit_table = AMRWB_IO_BITS if mode_bit else PRIMARY_BITS
            bit_length = bit_table.get(ft_index)
            if bit_length is None:
                raise ValueError(f"unsupported EVS frame type {ft}")
            if mode_bit:
                mode = "amr-wb-io"
                is_sid = ft_index == 9
            else:
                mode = "primary"
                is_sid = ft_index == 12

            valid = True
            speech = True
            if ft_index == 15:
                frame_type = "no_data"
                valid = False
                speech = False
            elif ft_index == 14:
                frame_type = "speech_lost"
                valid = False
                speech = False
            elif is_sid:
                frame_type = "sid"
                speech = False
            else:
                frame_type = "speech"

            byte_count = (bit_length + 7) // 8
            start = offset_bits // 8
            data = payload[start : start + byte_count] if bit_length else b""
            if bit_length and len(data) != byte_count:
                raise ValueError("truncated EVS speech frame")

            frames.append(
                EncodedFrame(
                    codec="EVS",
                    data=data,
                    rtp_timestamp=(
                        packet.timestamp + index * EVS_FRAME_SAMPLES
                    ) & 0xFFFFFFFF,
                    duration=EVS_FRAME_SAMPLES,
                    sample_rate=EVS_CLOCK_RATE,
                    valid=valid,
                    speech=speech,
                    frame_type=frame_type,
                    sequence=packet.sequence,
                    bit_offset=0,
                    bit_length=bit_length,
                    mode=mode,
                )
            )
            offset_bits += byte_count * 8

        return frames
```

- [ ] **Step 4: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_evs_depacketizer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bad_audio/io/rtp/depacketizer/evs.py tests/unit/test_evs_depacketizer.py
git commit -m "feat: parse EVS header-full payloads and AMR-WB IO"
```

---

## Task 5: Use a per-session depacketizer and pass EVS options through stream config

**Files:**
- Modify: `src/bad_audio/io/rtp/depacketizer/base.py`
- Modify: `src/bad_audio/io/rtp/depacketizer/__init__.py:37-48`
- Modify: `src/bad_audio/streaming/session.py:18-216`
- Test: `tests/unit/test_evs_specs.py`

**Interfaces:**
- Consumes: `get_depacketizer(spec)` and `EvsDepacketizer`.
- Produces:
  - `Depacketizer.flush() -> list[EncodedFrame]`
  - `StreamSession.depacketizer: Depacketizer | None`
  - `create_session` accepts EVS options: `hf_only`, `evs_mode`, `evs_bitrate`/`evs_bitrates`, `evs_bandwidth`/`evs_bandwidths`, `dtx`, `cmr`, `ch_aw_recv`, `max_red`, `interleaving`, `interleaving_max`.

- [ ] **Step 1: Write the failing session test**

Add to `tests/unit/test_evs_specs.py`:

```python
def test_create_session_evs_options():
    from bad_audio.config import Settings
    from bad_audio.pipeline import Detector
    from bad_audio.streaming.session import create_session

    settings = Settings(_env_file=None)
    detector = Detector(model=None, settings=settings)
    session = create_session(
        detector,
        settings,
        {
            "format": "rtp",
            "codec": "EVS",
            "clock_rate": 16000,
            "hf_only": 1,
            "evs_mode": "amr-wb-io",
            "dtx": 0,
            "cmr": 1,
            "br": "5.9-24.4",
            "bw": "nb-wb",
            "ch_aw_recv": 2,
        },
    )

    assert session.spec is not None
    assert session.spec.encoding == "EVS"
    assert session.spec.hf_only is True
    assert session.spec.evs_mode == "amr-wb-io"
    assert session.spec.evs_dtx is False
    assert session.spec.evs_cmr == 1
    assert session.spec.evs_bitrates == (5900.0, 24400.0)
    assert session.spec.evs_bandwidths == ("nb", "wb")
    assert session.spec.evs_channel_aware == 2
    assert session.depacketizer is not None
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_specs.py::test_create_session_evs_options -v
```

Expected: `AttributeError: 'StreamSession' object has no attribute 'depacketizer'`.

- [ ] **Step 3: Add `flush()` to the base depacketizer**

In `src/bad_audio/io/rtp/depacketizer/base.py`:

```python
class Depacketizer:
    def depacketize(self, packet, spec) -> list[EncodedFrame]:
        raise NotImplementedError

    def flush(self) -> list[EncodedFrame]:
        return []
```

- [ ] **Step 4: Make depacketizers per-session instances**

In `src/bad_audio/io/rtp/depacketizer/__init__.py`, replace `get_depacketizer` and `depacketize`:

```python
def get_depacketizer(spec: CodecSpec) -> Depacketizer:
    return _make(spec.encoding)


def depacketize(packet, spec: CodecSpec) -> list[EncodedFrame]:
    return get_depacketizer(spec).depacketize(packet, spec)
```

Remove `_CACHE` entirely.

- [ ] **Step 5: Give `StreamSession` a depacketizer and flush it**

In `src/bad_audio/streaming/session.py`:

```python
from ..io.rtp.depacketizer import (
    Depacketizer,
    EncodedFrame,
    get_depacketizer,
    make_spec,
)
```

Add the field to `StreamSession`:

```python
    depacketizer: Depacketizer | None = None
```

Replace both `depacketize(ready, self.spec)` calls:

```python
        for ready in self.jitter.drain():
            self.frames_buffer.extend(self.depacketizer.depacketize(ready, self.spec))
```

and in `finish`:

```python
        if (
            self.kind == "rtp"
            and self.jitter is not None
            and self.spec is not None
            and self.depacketizer is not None
        ):
            for ready in self.jitter.flush():
                self.frames_buffer.extend(self.depacketizer.depacketize(ready, self.spec))
            self.frames_buffer.extend(self.depacketizer.flush())
```

In `create_session`, after `spec = replace(spec, max_frames_per_packet=settings.max_frames_per_packet)`, add:

```python
    depacketizer = get_depacketizer(spec)
```

Pass `depacketizer=depacketizer` to the RTP `StreamSession` constructor.

Update `_require_rtp` so a missing depacketizer is reported early:

```python
    def _require_rtp(self) -> None:
        if (
            self.kind != "rtp"
            or self.spec is None
            or self.jitter is None
            or self.depacketizer is None
        ):
            raise ValueError("RTP session is not configured")
```

- [ ] **Step 6: Add EVS config passthrough**

In `src/bad_audio/streaming/session.py`, import the EVS helpers:

```python
from ..io.rtp.specs import (
    CodecSpec,
    parse_evs_bandwidths,
    parse_evs_bitrates,
    spec_from_pt,
)
```

In the `elif config.get("codec"):` branch, after the existing overrides, add:

```python
        if "hf_only" in config:
            overrides["hf_only"] = bool(int(config["hf_only"]))
        if "evs_mode" in config:
            overrides["evs_mode"] = str(config["evs_mode"])
        if "dtx" in config:
            overrides["evs_dtx"] = bool(int(config["dtx"]))
        if "cmr" in config:
            overrides["evs_cmr"] = int(config["cmr"])
        if "ch_aw_recv" in config:
            overrides["evs_channel_aware"] = int(config["ch_aw_recv"])
        if "max_red" in config:
            overrides["evs_max_red"] = int(config["max_red"])
        if "br" in config:
            overrides["evs_bitrates"] = parse_evs_bitrates(str(config["br"]))
        if "bw" in config:
            overrides["evs_bandwidths"] = parse_evs_bandwidths(str(config["bw"]))
        if "interleaving_max" in config:
            overrides["interleaving_max"] = int(config["interleaving_max"])
```

- [ ] **Step 7: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_evs_specs.py tests/unit/test_depacketizers.py tests/integration/test_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/bad_audio/io/rtp/depacketizer/base.py src/bad_audio/io/rtp/depacketizer/__init__.py src/bad_audio/streaming/session.py tests/unit/test_evs_specs.py
git commit -m "feat: add per-session depacketizers and EVS stream options"
```

---

## Task 6: Implement the EVS decoder backend

**Files:**
- Modify: `src/bad_audio/io/rtp/decoder.py:1-156`
- Modify: `src/bad_audio/pipeline.py:61-70`
- Create: `tests/fixtures/fake_evs_dec.py`
- Test: `tests/unit/test_evs_decoder.py`

**Interfaces:**
- Consumes: `Settings.evs_decoder_cmd`, `Settings.evs_decoder_rate`, `Settings.evs_decoder_timeout_s`; `EncodedFrame.bit_offset`, `bit_length`, `mode`.
- Produces:
  - `CodecDecoder(target_rate, evs_decoder_cmd, evs_decoder_rate, evs_decoder_timeout_s)`
  - `_decode_evs(frames, spec) -> np.ndarray`
  - `_g192_bytes(frames) -> bytes`
  - `Detector.decoder` configured from settings.

- [ ] **Step 1: Create the fake `EVS_dec` fixture**

Create `tests/fixtures/fake_evs_dec.py`:

```python
from __future__ import annotations

import os
import struct
import sys


def main() -> None:
    fs_khz = int(sys.argv[1])
    src = sys.argv[2]
    dst = sys.argv[3]

    data = open(src, "rb").read()
    offset = 0
    frames: list[tuple[int, int]] = []
    while offset + 4 <= len(data):
        sync, nbits = struct.unpack_from("<HH", data, offset)
        offset += 4
        frames.append((sync, nbits))
        offset += nbits * 2

    meta = os.environ.get("FAKE_EVS_META")
    if meta:
        with open(meta, "w", encoding="utf-8") as handle:
            for sync, nbits in frames:
                handle.write(f"{sync:04x}:{nbits}\n")

    pcm = b"".join(
        struct.pack("<h", 0) for _ in range(len(frames) * (fs_khz * 20))
    )
    with open(dst, "wb") as handle:
        handle.write(pcm)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing decoder tests**

Create `tests/unit/test_evs_decoder.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from bad_audio.io.rtp.decoder import CodecDecoder
from bad_audio.io.rtp.depacketizer import EncodedFrame, make_spec

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "fake_evs_dec.py"


def evs_frame(
    data: bytes,
    *,
    timestamp: int = 0,
    valid: bool = True,
    frame_type: str = "speech",
    bit_length: int = 160,
) -> EncodedFrame:
    return EncodedFrame(
        codec="EVS",
        data=data,
        rtp_timestamp=timestamp,
        duration=320,
        sample_rate=16000,
        valid=valid,
        speech=frame_type == "speech",
        frame_type=frame_type,
        bit_length=bit_length,
        mode="primary",
    )


def test_evs_decode_uses_reference_decoder(tmp_path, monkeypatch):
    meta = tmp_path / "meta.txt"
    monkeypatch.setenv("FAKE_EVS_META", str(meta))
    decoder = CodecDecoder(
        target_rate=16000,
        evs_decoder_cmd=f"{sys.executable} {FIXTURE}",
        evs_decoder_timeout_s=10.0,
    )

    frames = [
        evs_frame(bytes([0xAA]) * 20, timestamp=0),
        evs_frame(
            b"",
            timestamp=320,
            valid=False,
            frame_type="no_data",
            bit_length=0,
        ),
    ]
    decoded = decoder.decode(frames, make_spec("EVS"))

    assert decoded.pcm.dtype == np.float32
    assert decoded.pcm.shape == (640,)
    assert decoded.valid_mask[:320].all()
    assert not decoded.valid_mask[320:].any()

    lines = meta.read_text(encoding="utf-8").splitlines()
    # Invalid frames are represented by the decoder's valid mask, not by
    # passing them to EVS_dec, so only the valid frame appears in G.192.
    assert lines == ["6b21:160"]


def test_evs_decode_without_backend_raises():
    decoder = CodecDecoder(target_rate=16000, evs_decoder_cmd="")
    with pytest.raises(RuntimeError, match="EVS decoder"):
        decoder.decode([evs_frame(bytes([0xAA]) * 20)], make_spec("EVS"))


def test_evs_g192_writes_each_bit_msb_first():
    decoder = CodecDecoder(target_rate=16000, evs_decoder_cmd="")
    frame = evs_frame(bytes([0b1010_0000]), bit_length=4)
    payload = decoder._g192_bytes([frame])

    assert payload[:4] == b"\x21\x6b\x04\x00"
    words = [
        int.from_bytes(payload[index : index + 2], "little")
        for index in range(4, len(payload), 2)
    ]
    assert words == [0x0081, 0x007F, 0x0081, 0x007F]
```

- [ ] **Step 3: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_evs_decoder.py -v
```

Expected: `TypeError` because `CodecDecoder.__init__` does not accept `evs_decoder_cmd`.

- [ ] **Step 4: Implement G.192 writing and EVS decoding**

In `src/bad_audio/io/rtp/decoder.py`, add imports:

```python
import shlex
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
```

Import `read_bits`:

```python
from ..rtp.depacketizer.evs import read_bits
```

Change `CodecDecoder.__init__`:

```python
class CodecDecoder:
    def __init__(
        self,
        target_rate: int = 16000,
        evs_decoder_cmd: str = "EVS_dec",
        evs_decoder_rate: int = 16000,
        evs_decoder_timeout_s: float = 5.0,
    ) -> None:
        self.target_rate = target_rate
        self.evs_decoder_cmd = evs_decoder_cmd
        self.evs_decoder_rate = evs_decoder_rate
        self.evs_decoder_timeout_s = evs_decoder_timeout_s
```

Add the EVS branch in `_decode_group`:

```python
        if encoding == "EVS":
            return self._decode_evs(frames, spec)
```

Add methods after `_decode_group`:

```python
    def _g192_bytes(self, frames: list[EncodedFrame]) -> bytes:
        chunks: list[bytes] = []
        for frame in frames:
            if not frame.valid or frame.frame_type in ("no_data", "speech_lost"):
                chunks.append(struct.pack("<HH", 0x6B20, 0))
                continue
            bit_length = frame.bit_length or len(frame.data) * 8
            chunks.append(struct.pack("<HH", 0x6B21, bit_length))
            for bit in read_bits(frame.data, frame.bit_offset, bit_length):
                chunks.append(struct.pack("<H", 0x0081 if bit else 0x007F))
        return b"".join(chunks)

    def _decode_evs(self, frames: list[EncodedFrame], spec: CodecSpec) -> np.ndarray:
        command = shlex.split(self.evs_decoder_cmd)
        if not command:
            raise RuntimeError(
                "EVS decoder backend not configured; set EVS_DECODER_CMD"
            )
        executable = command[0]
        if shutil.which(executable) is None and not Path(executable).exists():
            raise RuntimeError(f"EVS decoder command not found: {executable}")

        with tempfile.TemporaryDirectory(prefix="evs-") as tmp:
            source = Path(tmp) / "input.192"
            output = Path(tmp) / "output.pcm"
            source.write_bytes(self._g192_bytes(frames))
            proc = subprocess.run(
                command
                + [
                    str(self.evs_decoder_rate // 1000),
                    str(source),
                    str(output),
                ],
                capture_output=True,
                timeout=self.evs_decoder_timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="ignore")[:500]
                raise RuntimeError(f"EVS decoder failed: {stderr}")
            if not output.exists():
                raise RuntimeError("EVS decoder produced no output file")
            raw = output.read_bytes()

        if not raw:
            raise RuntimeError("EVS decoder produced empty PCM")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return resample(pcm, self.evs_decoder_rate, self.target_rate)
```

- [ ] **Step 5: Pass settings from `Detector`**

In `src/bad_audio/pipeline.py`:

```python
        self.decoder = CodecDecoder(
            target_rate=self.settings.target_sample_rate,
            evs_decoder_cmd=self.settings.evs_decoder_cmd,
            evs_decoder_rate=self.settings.evs_decoder_rate,
            evs_decoder_timeout_s=self.settings.evs_decoder_timeout_s,
        )
```

- [ ] **Step 6: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_evs_decoder.py tests/unit/test_codec_decoder.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bad_audio/io/rtp/decoder.py src/bad_audio/pipeline.py tests/fixtures/fake_evs_dec.py tests/unit/test_evs_decoder.py
git commit -m "feat: decode EVS via 3GPP EVS_dec G.192 backend"
```

---

## Task 7: Add end-to-end EVS RTP WebSocket test

**Files:**
- Modify: `tests/integration/test_api.py`
- Uses: `tests/fixtures/fake_evs_dec.py`

**Interfaces:**
- Consumes: `Detector(model=None, settings=Settings(evs_decoder_cmd=...))`, `create_session` EVS config, `EvsDepacketizer`, `CodecDecoder._decode_evs`.
- Produces: proof that EVS RTP can be configured and decoded through the public WebSocket API.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/integration/test_api.py`:

```python
def test_websocket_rtp_evs_stream():
    import sys
    from pathlib import Path

    from bad_audio.config import Settings

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "fake_evs_dec.py"
    settings = Settings(
        _env_file=None,
        evs_decoder_cmd=f"{sys.executable} {fixture}",
        evs_decoder_timeout_s=10.0,
    )
    app.state.detector = Detector(model=None, settings=settings)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json(
                {
                    "type": "config",
                    "format": "rtp",
                    "payload_type": 96,
                    "codec": "EVS",
                    "clock_rate": 16000,
                }
            )
            configured = ws.receive_json()
            assert configured["type"] == "configured"
            assert configured["codec"] == "EVS"

            for index in range(100):
                ws.send_bytes(make_rtp(bytes(20), seq=index, ts=index * 320, pt=96))
            ws.send_json({"type": "eos"})

            while True:
                message = ws.receive_json()
                if message["type"] in ("verdict", "error"):
                    break

            assert message["type"] == "verdict", message
            assert message["insufficient_signal"] is True
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
python3 -m pytest tests/integration/test_api.py::test_websocket_rtp_evs_stream -v
```

Expected: it fails before the EVS decoder wiring; after Task 6 it should pass.

- [ ] **Step 3: Verify the test passes**

Run:

```bash
python3 -m pytest tests/integration/test_api.py::test_websocket_rtp_evs_stream -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_api.py
git commit -m "test: cover EVS RTP WebSocket streaming"
```

---

## Task 8: Add AMR mode-set enforcement and robust-sorting support

**Files:**
- Modify: `src/bad_audio/io/rtp/depacketizer/amr.py:64-181`
- Modify: `tests/unit/test_depacketizers.py`

**Interfaces:**
- Consumes: `CodecSpec.mode_set`, `CodecSpec.robust_sorting`.
- Produces:
  - AMR/AMR-WB frame types outside a negotiated `mode_set` raise `ValueError`.
  - `robust_sorting=True` reconstructs round-robin octet-interleaved speech frames before decoding.

- [ ] **Step 1: Write the failing mode-set test**

Add to `tests/unit/test_depacketizers.py`:

```python
def test_amr_wb_mode_set_rejects_disallowed_speech_frame():
    spec = make_spec("AMR-WB", octet_align=True, mode_set=(0, 1))
    payload = bytes([0x00, 0x2C]) + bytes(46)
    with pytest.raises(ValueError, match="mode-set"):
        depacketize(rtp(payload, pt=96), spec)


def test_amr_wb_mode_set_allows_sid():
    spec = make_spec("AMR-WB", octet_align=True, mode_set=(0, 1))
    payload = bytes([0x00, (9 << 3) | 0x04]) + bytes(5)
    frames = depacketize(rtp(payload, pt=96), spec)
    assert len(frames) == 1
    assert frames[0].frame_type == "sid"
```

- [ ] **Step 2: Write the failing robust-sorting test**

Add to `tests/unit/test_depacketizers.py`:

```python
def test_amr_oa_robust_sorting_reconstructs_frames():
    frame1 = bytes(range(31))
    frame2 = bytes(range(100, 131))
    data = bytearray()
    for index in range(31):
        data.append(frame1[index])
        data.append(frame2[index])

    payload = bytes([0xF0, 0xBC, 0x3C]) + bytes(data)
    frames = depacketize(
        rtp(payload, pt=98),
        make_spec("AMR-NB", octet_align=True, robust_sorting=True),
    )

    assert len(frames) == 2
    assert frames[0].data[1:] == frame1
    assert frames[1].data[1:] == frame2
```

- [ ] **Step 3: Run them and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_depacketizers.py -k "mode_set or robust_sorting" -v
```

Expected: mode-set test does not raise; robust-sorting test gets interleaved bytes in the wrong order.

- [ ] **Step 4: Add mode-set enforcement**

In `src/bad_audio/io/rtp/depacketizer/amr.py`, change `_classify` to accept `spec`:

```python
    def _classify(self, ft: int, spec) -> tuple[int, bool, bool, str]:
        if ft not in self.bits:
            raise ValueError(f"invalid AMR frame type {ft}")
        nbits = self.bits[ft]
        sid_ft = 9 if self.wideband else 8
        if ft == 15:
            return 0, False, False, "no_data"
        if ft == 14:
            if not self.wideband:
                raise ValueError("invalid AMR-NB frame type 14")
            return 0, False, False, "speech_lost"
        if ft == sid_ft:
            return nbits, True, False, "sid"
        if spec.mode_set is not None and ft not in spec.mode_set:
            raise ValueError(f"AMR frame type {ft} is outside the negotiated mode-set")
        return nbits, True, True, "speech"
```

Update all `self._classify(ft)` calls to `self._classify(ft, spec)`.

- [ ] **Step 5: Add robust-sorting reconstruction**

In `src/bad_audio/io/rtp/depacketizer/amr.py`, add helper:

```python
    def _read_octet_frames(
        self,
        payload: bytes,
        offset: int,
        lengths: list[int],
        robust: bool,
    ) -> tuple[list[bytes], int]:
        if not robust:
            data: list[bytes] = []
            for length in lengths:
                raw = payload[offset : offset + length]
                if len(raw) != length:
                    raise ValueError("truncated AMR speech frame")
                data.append(raw)
                offset += length
            return data, offset

        buffers = [bytearray(length) for length in lengths]
        max_length = max(lengths, default=0)
        for octet_index in range(max_length):
            for frame_index, length in enumerate(lengths):
                if octet_index < length:
                    if offset >= len(payload):
                        raise ValueError("truncated AMR speech frame")
                    buffers[frame_index][octet_index] = payload[offset]
                    offset += 1
        return [bytes(buffer) for buffer in buffers], offset
```

In `_parse_octet_aligned`, replace the per-entry data-reading loop with:

```python
        lengths = [(nbits + 7) // 8 if nbits else 0 for nbits, _, _, _ in classified]
        raws, offset = self._read_octet_frames(
            payload,
            offset,
            lengths,
            spec.robust_sorting,
        )

        entries = []
        for (ft, quality), (nbits, valid, speech, frame_type), raw in zip(
            toc, classified, raws
        ):
            entries.append(
                (ft, quality, raw if nbits else None, valid, speech, frame_type)
            )
        return entries, None
```

- [ ] **Step 6: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_depacketizers.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bad_audio/io/rtp/depacketizer/amr.py tests/unit/test_depacketizers.py
git commit -m "feat: support AMR mode sets and robust sorting"
```

---

## Task 9: Add AMR frame-interleaving support

**Files:**
- Modify: `src/bad_audio/io/rtp/depacketizer/amr.py`
- Modify: `src/bad_audio/streaming/session.py` if `interleaving_max` passthrough was not added in Task 5
- Modify: `tests/unit/test_depacketizers.py`
- Modify: `tests/unit/test_evs_specs.py` if a session interleaving test is needed

**Interfaces:**
- Consumes: `CodecSpec.interleaving`, `CodecSpec.interleaving_max`.
- Produces:
  - `AmrDepacketizer` parses the optional ILL/ILP header.
  - Frame timestamps use the interleaving stride `(ILL + 1) * frame_samples`.
  - Frames are buffered per interleaving group and emitted in timestamp order when complete.
  - `AmrDepacketizer.flush()` emits any incomplete groups.

- [ ] **Step 1: Replace the old rejection test with a reordering test**

Remove `test_amr_interleaving_is_rejected` from `tests/unit/test_depacketizers.py` and add:

```python
def test_amr_oa_interleaving_reorders_frames():
    from bad_audio.io.rtp.depacketizer.amr import AmrDepacketizer

    frame1 = bytes(range(31))
    frame2 = bytes(range(100, 131))
    frame3 = bytes(range(31))
    frame4 = bytes(range(100, 131))

    payload0 = bytes([0xF0, 0x10, 0xBC, 0x3C]) + frame1 + frame2
    payload1 = bytes([0xF0, 0x11, 0xBC, 0x3C]) + frame3 + frame4

    spec = make_spec(
        "AMR-NB",
        octet_align=True,
        interleaving=True,
        interleaving_max=4,
    )
    depacketizer = AmrDepacketizer(wideband=False)

    assert depacketizer.depacketize(rtp(payload0, pt=98, ts=1000), spec) == []

    frames = depacketizer.depacketize(rtp(payload1, pt=98, ts=1160), spec)
    assert [frame.rtp_timestamp for frame in frames] == [1000, 1160, 1320, 1480]
    assert frames[0].data[1:] == frame1
    assert frames[1].data[1:] == frame3
    assert frames[2].data[1:] == frame2
    assert frames[3].data[1:] == frame4


def test_amr_interleaving_flush_emits_partial_group():
    from bad_audio.io.rtp.depacketizer.amr import AmrDepacketizer

    payload = bytes([0xF0, 0x10, 0xBC, 0x3C]) + bytes(62)
    spec = make_spec(
        "AMR-NB",
        octet_align=True,
        interleaving=True,
        interleaving_max=4,
    )
    depacketizer = AmrDepacketizer(wideband=False)

    assert depacketizer.depacketize(rtp(payload, pt=98, ts=1000), spec) == []
    frames = depacketizer.flush()
    assert len(frames) == 2
```

- [ ] **Step 2: Run them and verify failure**

Run:

```bash
python3 -m pytest tests/unit/test_depacketizers.py -k "interleaving" -v
```

Expected: the old rejection test is gone; the new tests fail because interleaving is still rejected.

- [ ] **Step 3: Add interleaving state to `AmrDepacketizer`**

In `src/bad_audio/io/rtp/depacketizer/amr.py`, update `__init__`:

```python
    def __init__(self, wideband: bool) -> None:
        self.wideband = wideband
        self.bits = AMR_WB_BITS if wideband else AMR_NB_BITS
        self.frame_samples = self.WB_FRAME_SAMPLES if wideband else self.NB_FRAME_SAMPLES
        self.codec = "AMR-WB" if wideband else "AMR-NB"
        self._groups: dict[tuple[int, int], dict] = {}
```

Add `flush`:

```python
    def flush(self) -> list[EncodedFrame]:
        frames = []
        for group in self._groups.values():
            frames.extend(group["frames"][ts] for ts in sorted(group["frames"]))
        self._groups.clear()
        return frames
```

- [ ] **Step 4: Parse the interleaving header and build frame timestamps**

Replace `depacketize` with:

```python
    def depacketize(self, packet, spec) -> list[EncodedFrame]:
        if spec.channels != 1:
            raise NotImplementedError("multichannel AMR depacketization is not supported yet")
        if spec.robust_sorting and not spec.octet_align:
            raise ValueError("AMR robust sorting requires octet-aligned mode")
        if not packet.payload:
            return []

        interleave_ill = 0
        interleave_ilp = 0
        if spec.interleaving:
            if not spec.octet_align:
                raise ValueError("AMR interleaving requires octet-aligned mode")
            if len(packet.payload) < 2:
                raise ValueError("truncated AMR interleaving header")
            interleave_ill = (packet.payload[1] >> 4) & 0x0F
            interleave_ilp = packet.payload[1] & 0x0F
            if interleave_ilp > interleave_ill:
                raise ValueError("invalid AMR interleaving index")
            if spec.interleaving_max and interleave_ill + 1 > spec.interleaving_max:
                raise ValueError("AMR interleaving length exceeds negotiated maximum")

        entries, frame_data = (
            self._parse_octet_aligned(
                packet.payload,
                spec,
                interleave=spec.interleaving,
            )
            if spec.octet_align
            else self._parse_bandwidth_efficient(packet.payload, spec)
        )

        step = self.frame_samples * (interleave_ill + 1)
        frames: list[EncodedFrame] = []
        for index, (ft, quality, raw, valid, speech, frame_type) in enumerate(entries):
            header = bytes([(ft << 3) | (quality << 2)]) if raw is not None else b""
            frames.append(
                EncodedFrame(
                    codec=self.codec,
                    data=header + (raw or b""),
                    rtp_timestamp=(packet.timestamp + index * step) & 0xFFFFFFFF,
                    duration=self.frame_samples,
                    sample_rate=spec.sample_rate,
                    valid=valid,
                    speech=speech,
                    frame_type=frame_type,
                    sequence=packet.sequence,
                )
            )

        if not spec.interleaving:
            return frames

        group_start = (packet.timestamp - interleave_ilp * self.frame_samples) & 0xFFFFFFFF
        key = (packet.ssrc, group_start)
        group = self._groups.get(key)
        if group is None:
            group = {
                "frames": {},
                "size": len(frames) * (interleave_ill + 1),
            }
            self._groups[key] = group
        elif group["size"] != len(frames) * (interleave_ill + 1):
            raise ValueError("inconsistent AMR interleaving group size")

        for frame in frames:
            group["frames"].setdefault(frame.rtp_timestamp, frame)

        if len(group["frames"]) < group["size"]:
            return []

        emitted = [group["frames"][ts] for ts in sorted(group["frames"])]
        del self._groups[key]
        return emitted
```

- [ ] **Step 5: Update `_parse_octet_aligned` for the interleaving header**

Change the signature:

```python
    def _parse_octet_aligned(self, payload: bytes, spec, interleave: bool = False):
```

Change the initial offset:

```python
        offset = 2 if interleave else 1
```

The rest of the function uses `offset` as before.

- [ ] **Step 6: Run the tests**

Run:

```bash
python3 -m pytest tests/unit/test_depacketizers.py tests/unit/test_evs_specs.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/bad_audio/io/rtp/depacketizer/amr.py src/bad_audio/streaming/session.py tests/unit/test_depacketizers.py tests/unit/test_evs_specs.py
git commit -m "feat: support AMR frame interleaving"
```

---

## Task 10: Install the EVS decoder in Docker and update documentation

**Files:**
- Modify: `Dockerfile:1-40`
- Modify: `.env.example`
- Modify: `README.md:573-593`, `README.md:680-715`

**Interfaces:**
- Consumes: `EVS_DECODER_CMD` setting handled by `CodecDecoder`.
- Produces: a Docker image with `/usr/local/bin/EVS_dec`, and documentation for EVS/AMR settings and troubleshooting.

- [ ] **Step 1: Add the EVS decoder build stage**

At the top of `Dockerfile`, add:

```dockerfile
FROM python:3.10-slim AS evs-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG EVS_CODEC_REPO=https://github.com/wanglihe/3gpp-evs.git
ARG EVS_CODEC_REF=master

RUN git clone --depth 1 --branch "${EVS_CODEC_REF}" "${EVS_CODEC_REPO}" /src/evs \
    && make -C /src/evs/c-code \
    && install -m 0755 /src/evs/c-code/EVS_dec /usr/local/bin/EVS_dec
```

Then change the existing first line from:

```dockerfile
FROM python:3.10-slim
```

to:

```dockerfile
FROM python:3.10-slim AS runtime
```

Add to the runtime stage after the existing `ENV` block:

```dockerfile
ENV EVS_DECODER_CMD=EVS_dec \
    EVS_DECODER_RATE=16000 \
    EVS_DECODER_TIMEOUT_S=5.0
```

Add after the `RUN apt-get ...` runtime install block:

```dockerfile
COPY --from=evs-builder /usr/local/bin/EVS_dec /usr/local/bin/EVS_dec
```

- [ ] **Step 2: Verify the image builds**

Run:

```bash
docker build -t bad-audio-detector:evs .
docker run --rm --entrypoint ls bad-audio-detector:evs -l /usr/local/bin/EVS_dec
```

Expected: the build succeeds and `ls` lists `/usr/local/bin/EVS_dec`.

If the default repository or branch is unavailable, override it with another
mirror, for example:

```bash
docker build \
  --build-arg EVS_CODEC_REPO=https://github.com/old6ix/EVS_codec.git \
  --build-arg EVS_CODEC_REF=master \
  -t bad-audio-detector:evs .
```

- [ ] **Step 3: Update the README support matrix**

Replace the AMR/AMR-WB rows and append EVS in `README.md` section 6:

```markdown
| AMR-NB | 8000 | 8000 | 20 ms | yes | All modes 0-8, BE/OA, CMR/TOC, CRC, SID/NO_DATA, robust sorting, interleaving |
| AMR-WB | 16000 | 16000 | 20 ms | yes | All modes 0-9, BE/OA, CMR/TOC, CRC, SID/SPEECH_LOST/NO_DATA, robust sorting, interleaving |
| EVS | 16000 | 8/16/32/48k→16k | 20 ms | yes | EVS Primary 5.9-128 kbps + AMR-WB IO 6.6-23.85 kbps, Compact/Header-Full, DTX/SID, partial redundancy. Requires `EVS_dec`. |
```

Replace the paragraph about AMR/G.729 encoders with:

```markdown
AMR-NB, AMR-WB, and EVS are decode-only at runtime. The Docker image builds
the 3GPP reference `EVS_dec` decoder and sets `EVS_DECODER_CMD=EVS_dec`.
Stock FFmpeg/PyAV does not include an EVS decoder. Synthetic dataset generation
still only uses codecs that the local PyAV build can encode.
```

Add an EVS SDP example near the existing RTP examples:

````markdown
EVS Primary with Header-Full-only and DTX:

```json
{"type":"config","format":"rtp","sdp":"v=0\r\nm=audio 49170 RTP/AVP 96\r\na=rtpmap:96 EVS/16000/1\r\na=fmtp:96 hf-only=1; dtx=1; br=5.9-24.4; bw=nb-wb\r\n"}
```

EVS AMR-WB IO mode:

```json
{"type":"config","format":"rtp","sdp":"v=0\r\nm=audio 49170 RTP/AVP 96\r\na=rtpmap:96 EVS/16000/1\r\na=fmtp:96 evs-mode-switch=1; mode-set=0,1,2\r\n"}
```
````

Add troubleshooting entries in `README.md` section 11:

```markdown
- **EVS decode fails:** confirm `EVS_dec` is installed and executable, or set
  `EVS_DECODER_CMD` to the absolute path of the reference decoder. The stock
  PyAV/FFmpeg wheel has no EVS decoder.
- **EVS decoder times out:** increase `EVS_DECODER_TIMEOUT_S`; the reference
  decoder is a subprocess and is slower than native FFmpeg decoders.
- **AMR interleaving looks out of order:** ensure the stream is configured
  with `interleaving=1` and, if needed, `interleaving_max=4`.
```

- [ ] **Step 4: Run documentation sanity checks**

Run:

```bash
grep -n "EVS" README.md
grep -n "EVS_DECODER" .env.example Dockerfile
```

Expected: EVS appears in the support matrix, SDP examples, troubleshooting, `.env.example`, and `Dockerfile`.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .env.example README.md
git commit -m "docs: document EVS support and install EVS_dec in Docker"
```

---

## Task 11: Full verification and cleanup

**Files:**
- No new files.

- [ ] **Step 1: Run the full unit test suite**

```bash
python3 -m pytest tests/unit -v
```

Expected: all unit tests pass.

- [ ] **Step 2: Run integration tests**

```bash
python3 -m pytest tests/integration -v
```

Expected: all integration tests pass.

- [ ] **Step 3: Run the complete suite**

```bash
python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Run lint**

```bash
python3 -m ruff check src tests scripts
```

Expected: no lint errors.

- [ ] **Step 5: Check the Docker image**

```bash
docker build -t bad-audio-detector:evs .
docker run --rm --entrypoint python bad-audio-detector:evs -c "import av; print(av.__version__)"
docker run --rm --entrypoint ls bad-audio-detector:evs -l /usr/local/bin/EVS_dec
```

Expected: build succeeds, PyAV imports, and `EVS_dec` exists.

- [ ] **Step 6: Final commit**

```bash
git add src tests Dockerfile .env.example README.md docs/superpowers/plans/2026-11-13-evs-amr-codec-support.md
git commit -m "test: verify EVS and AMR codec support"
```

---

## Out of Scope / Follow-up

- **EVS/AMR-WB synthetic encoding:** this plan adds runtime decode only. The synthetic dataset generator still uses PyAV-encodable codecs. To train on EVS/AMR-WB, add an encoder backend or collect real recordings.
- **Model accuracy:** the shipped model was not trained on EVS or AMR-WB. After adding data, retrain with `scripts/train.py` and compare per-codec metrics.
- **EVS AMR-WB IO over RFC 4867 format:** use codec `AMR-WB` for RFC 4867
  payloads. An `EVS` payload type that carries raw RFC 4867 format is not
  auto-detected; that requires a separate negotiation signal not covered here.
- **EVS channel-aware partial redundancy:** the depacketizer passes the EVS frame bits to the reference decoder; verifying channel-aware FEC behavior requires real channel-aware test vectors.
- **Multiple EVS mono channels:** explicitly rejected with `NotImplementedError`; add only if multi-channel EVS is required.

---

## Self-Review Notes

- Every CodecSpec field used in later tasks is introduced in Task 1.
- `EncodedFrame.bit_offset`, `bit_length`, and `mode` are introduced in Task 3 before the decoder uses them in Task 6.
- `get_depacketizer()` changes from cached to per-session in Task 5; the AMR interleaving state in Task 9 relies on that.
- EVS register names and SDP values match TS 26.445 Annex A: `evs-mode-switch`, `hf-only`, `dtx`, `cmr`, `max-red`, `br`, `bw`, `ch-aw-recv`, `mode-set`.
- AMR robust sorting is octet-level round-robin only and is rejected for bandwidth-efficient mode, matching RFC 4867.
- AMR interleaving timestamps use the stride `(ILL + 1) * frame_samples`, matching RFC 4867 section 4.4.1.
