from __future__ import annotations
import json
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BufDir(str, Enum):
    IN    = "in"
    OUT   = "out"
    INOUT = "inout"


OUT   = BufDir.OUT
IN    = BufDir.IN
INOUT = BufDir.INOUT

# Scalar types that fit in one uint32 args[] slot.
_PACK_FMT = {
    "i8":  "b", "u8":  "B",
    "i16": "h", "u16": "H",
    "i32": "i", "u32": "I",
    "f32": "f",
}


@dataclass
class ScalarArg:
    type:  str
    value: int

    def pack_u32(self) -> int:
        """Return value bit-cast to uint32 (little-endian, zero-extended).

        Narrow signed types (i8, i16) are zero-extended, not sign-extended.
        The device reads args[] as uint32 and casts to the target type itself,
        so the upper bytes are ignored; zero-extension is correct here.
        """
        raw = struct.pack(f"<{_PACK_FMT[self.type]}", self.value).ljust(4, b"\x00")
        return struct.unpack("<I", raw)[0]


@dataclass
class BufferArg:
    size: int                  # bytes; must be 4-byte aligned
    dir:  BufDir
    init: Optional[bytes] = None  # written to backing memory before kernel launch (IN/INOUT)


class SharedBuffer:
    """Physical buffer shared across multiple kernels in a LAUNCHES group.

    One physical allocation backs all kernels that reference this object.
    Allocated when the first referencing kernel is processed; freed only after
    ALL referencing kernels complete (ref-counted in Mmu_segment_manager).

    Optional host init: if provided, TestDriver writes it to the physical backing
    once (at allocation time) before any kernel launches. Useful when the buffer
    is an input read by multiple kernels.

    Flat data only: pointer-valued fields cannot be shared (logical address spaces
    are per-kernel; only the physical backing is common).

    TestDriver assigns a segment ID per kernel independently, so the logical
    address of the shared buffer may differ between kernels. The kernel always
    receives the correct address through its args[] slot.
    """

    _counter = 0

    def __init__(self, size: int, init: Optional[bytes] = None):
        if size % 4 != 0:
            raise ValueError(f"SharedBuffer size {size} must be 4-byte aligned")
        if init is not None and len(init) > size:
            raise ValueError(f"SharedBuffer init length {len(init)} exceeds size {size}")
        self.size = size
        self.init = init
        self._shared_id = f"sb_{SharedBuffer._counter}"
        SharedBuffer._counter += 1

    def as_input(self) -> SharedBufferView:
        return SharedBufferView(shared=self, dir=IN)

    def as_output(self) -> SharedBufferView:
        return SharedBufferView(shared=self, dir=OUT)

    def as_inout(self) -> SharedBufferView:
        return SharedBufferView(shared=self, dir=INOUT)


@dataclass
class SharedBufferView:
    """One kernel's directional view of a SharedBuffer."""
    shared: SharedBuffer
    dir:    BufDir


def scalar(type_str: str, value) -> ScalarArg:
    if type_str not in _PACK_FMT:
        raise ValueError(f"Unsupported scalar type '{type_str}'. Valid: {list(_PACK_FMT)}")
    return ScalarArg(type=type_str, value=value)


def buffer(size: int, dir: BufDir, init: Optional[bytes] = None) -> BufferArg:
    if size % 4 != 0:
        raise ValueError(f"Buffer size {size} must be 4-byte aligned")
    if init is not None and len(init) > size:
        raise ValueError(f"init length {len(init)} exceeds buffer size {size}")
    return BufferArg(size=size, dir=dir, init=init)


@dataclass
class KernelLaunch:
    """
    Kernel launch descriptor: host-side specification for a single kernel invocation.

    Use in a LAUNCHES list for multi-kernel launches:

        result = SharedBuffer(256)

        LAUNCHES = [
            KernelLaunch(elf="1CR/A/A.elf", grid=(1, 1, 4), args=[
                scalar("i32", 10),
                result.as_output(),
            ]),
            KernelLaunch(elf="1CR/B/B.elf", grid=(2, 1, 4), args=[
                result.as_input(),
                buffer(128, OUT),
            ]),
        ]

    elf:  absolute path to kernel ELF.
    grid: (n_x, n_y, n_ces): CR grid dimensions and CEs per CR.
    args: list of ScalarArg, BufferArg, or SharedBufferView, in declaration order.

    Buffer addresses and memory layout are assigned by the host driver consuming
    this descriptor; they are not part of the descriptor itself.
    """
    elf:  str
    grid: tuple   # (n_x, n_y, n_ces)
    args: list    # list[ScalarArg | BufferArg | SharedBufferView]

    def to_dict(self) -> dict:
        def _encode(a):
            if isinstance(a, ScalarArg):
                return {"kind": "scalar", "type": a.type, "value": a.value}
            if isinstance(a, BufferArg):
                return {
                    "kind": "buffer",
                    "size": a.size,
                    "dir":  a.dir.value,
                    "init": list(a.init) if a.init is not None else None,
                }
            if isinstance(a, SharedBufferView):
                return {
                    "kind":      "shared_buffer",
                    "shared_id": a.shared._shared_id,
                    "size":      a.shared.size,
                    "dir":       a.dir.value,
                    "init":      list(a.shared.init) if a.shared.init is not None else None,
                }
            raise TypeError(f"Unknown arg type: {type(a)}")

        return {
            "elf":  self.elf,
            "grid": list(self.grid),
            "args": [_encode(a) for a in self.args],
        }

    def to_json(self) -> str:
        return json.dumps({"kernels": [self.to_dict()]}, indent=2)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


def save_launches(launches: list, path: str) -> None:
    """Serialize a LAUNCHES list to a multi-kernel JSON file.

    Segment ID assignment is deferred to TestDriver. This function only
    serializes the semantic descriptor (elf, grid, arg kinds, shared_ids).
    """
    kernels = [launch.to_dict() for launch in launches]
    with open(path, "w") as f:
        json.dump({"kernels": kernels}, f, indent=2)
