from __future__ import annotations
import json
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Sub-directory (next to the launches JSON) holding raw init-data blobs.
# Buffer init bytes are large relative to scalar JSON fields; embedding them
# as JSON int arrays bloats the file and forces a slow element-by-element
# parse into std::vector<uint8_t> on the C++ side. Writing them as sibling
# .bin files lets the C++ loader mmap and hand the pointer straight to
# memif.write() -- no intermediate copy.
BUF_INIT_DIRNAME = "buf_init"


def _buf_dir_for(json_path: str) -> Path:
    return Path(json_path).parent / BUF_INIT_DIRNAME


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


# C type used for the local variable in generated redefine_main.c.
_CTYPE = {
    "i8":  "int8_t",  "u8":  "uint8_t",
    "i16": "int16_t", "u16": "uint16_t",
    "i32": "int32_t", "u32": "uint32_t",
    "f32": "float",
}


@dataclass
class ScalarArg:
    type:  str
    value: int
    name:  Optional[str] = None  # kernel param name; used by gen_redefine_main.py

    @property
    def ctype(self) -> str:
        return _CTYPE[self.type]

    def pack_u32(self) -> int:
        """Return value bit-cast to uint32 (little-endian, zero-extended).

        Narrow signed types (i8, i16) are zero-extended, not sign-extended.
        The device reads args[] as uint32 and casts to the target type itself,
        so the upper bytes are ignored; zero-extension is correct here.
        """
        raw = struct.pack(f"<{_PACK_FMT[self.type]}", self.value).ljust(4, b"\x00")
        return struct.unpack("<I", raw)[0]


def _validate_buffer(label: str, size: int, init: Optional[bytes]) -> None:
    if size % 4 != 0:
        raise ValueError(f"{label} size {size} must be 4-byte aligned")
    if init is not None and len(init) > size:
        raise ValueError(f"{label} init length {len(init)} exceeds size {size}")


@dataclass
class BufferArg:
    size: int                  # bytes; must be 4-byte aligned
    dir:  BufDir
    init: Optional[bytes] = None  # written to backing memory before kernel launch (IN/INOUT)
    name: Optional[str] = None    # kernel param name; used by gen_redefine_main.py
                                   # (pointee type is opaque -- generated as void *, like
                                   # OpenCL clCreateBuffer; only size is known here)


class SharedBuffer:
    """Physical buffer shared across multiple kernels in a LAUNCHES group.

    One physical allocation backs all kernels that reference this object.
    Allocated when the first referencing kernel is processed; freed only after
    ALL referencing kernels complete (ref-counted by the host driver).

    Optional host init: if provided, the host driver writes it to the physical
    backing once (at allocation time) before any kernel launches. Useful when
    the buffer is an input read by multiple kernels.

    Flat data only: pointer-valued fields cannot be shared (logical address spaces
    are per-kernel; only the physical backing is common).

    The host driver assigns each kernel its own address for the shared buffer
    independently, so that address may differ between kernels. The kernel
    always receives the correct address through its args[] slot.
    """

    _counter = 0

    def __init__(self, size: int, init: Optional[bytes] = None):
        _validate_buffer("SharedBuffer", size, init)
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


def scalar(type_str: str, value, name: Optional[str] = None) -> ScalarArg:
    if type_str not in _PACK_FMT:
        raise ValueError(f"Unsupported scalar type '{type_str}'. Valid: {list(_PACK_FMT)}")
    return ScalarArg(type=type_str, value=value, name=name)


def buffer(size: int, dir: BufDir, init: Optional[bytes] = None,
           name: Optional[str] = None) -> BufferArg:
    _validate_buffer("Buffer", size, init)
    return BufferArg(size=size, dir=dir, init=init, name=name)


def _bin_path(filename: str) -> str:
    return f"{BUF_INIT_DIRNAME}/{filename}"


def _write_init(buf_dir: Path, data: bytes, filename: str) -> str:
    buf_dir.mkdir(parents=True, exist_ok=True)
    (buf_dir / filename).write_bytes(data)
    return _bin_path(filename)


def _encode_scalar(a: ScalarArg) -> dict:
    return {"kind": "scalar", "type": a.type, "value": a.value, "name": a.name}


def _encode_buffer(a: BufferArg, buf_dir: Path, kernel_idx: int, arg_idx: int) -> dict:
    init_path = None
    if a.init is not None:
        init_path = _write_init(buf_dir, a.init, f"k{kernel_idx}_a{arg_idx}.bin")
    return {
        "kind": "buffer",
        "size": a.size,
        "dir":  a.dir.value,
        "init": init_path,
        "name": a.name,
    }


def _encode_shared_buffer_view(a: SharedBufferView, buf_dir: Path, shared_written: set) -> dict:
    shared_id = a.shared._shared_id
    init_path = None
    if a.shared.init is not None:
        filename = f"{shared_id}.bin"
        if shared_id not in shared_written:
            _write_init(buf_dir, a.shared.init, filename)
            shared_written.add(shared_id)
        init_path = _bin_path(filename)
    return {
        "kind":      "shared_buffer",
        "shared_id": shared_id,
        "size":      a.shared.size,
        "dir":       a.dir.value,
        "init":      init_path,
    }


@dataclass
class KernelLaunch:
    """
    Kernel launch descriptor: host-side specification for a single kernel invocation.

    Use in a LAUNCHES list for multi-kernel launches:

        result = SharedBuffer(256)

        LAUNCHES = [
            KernelLaunch(elf="1CR/A/A.elf", entry="aStart", grid=(1, 1, 4), args=[
                scalar("i32", 10, name="n"),
                result.as_output(),
            ]),
            KernelLaunch(elf="1CR/B/B.elf", entry="bStart", grid=(2, 1, 4), args=[
                result.as_input(),
                buffer(128, OUT, name="out"),
            ]),
        ]

    elf:   absolute path to kernel ELF.
    entry: kernel entry function name (e.g. "fibStart"); used by gen_redefine_main.py
           to emit the extern declaration and call in redefine_main.c. Return type is
           always void, like an OpenCL kernel.
    grid:  (n_x, n_y, n_ces): CR grid dimensions and CEs per CR.
    args:  list of ScalarArg, BufferArg, or SharedBufferView, in declaration order.

    Buffer addresses and memory layout are assigned by the host driver consuming
    this descriptor; they are not part of the descriptor itself.
    """
    elf:   str
    entry: str
    grid:  tuple   # (n_x, n_y, n_ces)
    args:  list    # list[ScalarArg | BufferArg | SharedBufferView]

    def to_dict(self, buf_dir: Path, kernel_idx: int = 0, shared_written: Optional[set] = None) -> dict:
        """Encode this launch to a JSON-able dict.

        Any buffer `init` bytes are written out to sibling .bin files under
        buf_dir rather than inlined as JSON int arrays (see BUF_INIT_DIRNAME).
        shared_written tracks which SharedBuffer ids already had their .bin
        written, so a buffer shared across multiple kernels in one
        save_launches() call is only written once.
        """
        if shared_written is None:
            shared_written = set()

        def _encode(arg_idx: int, a):
            if isinstance(a, ScalarArg):
                return _encode_scalar(a)
            if isinstance(a, BufferArg):
                return _encode_buffer(a, buf_dir, kernel_idx, arg_idx)
            if isinstance(a, SharedBufferView):
                return _encode_shared_buffer_view(a, buf_dir, shared_written)
            raise TypeError(f"Unknown arg type: {type(a)}")

        return {
            "elf":   self.elf,
            "entry": self.entry,
            "grid":  list(self.grid),
            "args":  [_encode(i, a) for i, a in enumerate(self.args)],
        }

    def to_json(self, buf_dir: Path, kernel_idx: int = 0) -> str:
        return json.dumps({"kernels": [self.to_dict(buf_dir, kernel_idx=kernel_idx)]}, indent=2)

    def save(self, path: str, kernel_idx: int = 0) -> None:
        """kernel_idx: distinguishes buf_init/ filenames when saving multiple
        KernelLaunch objects into the same directory (default 0 is fine for
        a single launch per directory; pass distinct values otherwise to
        avoid buffer-arg .bin filename collisions).

        Note: unlike save_launches(), this does not clear buf_dir first --
        repeated calls may target the same directory (see kernel_idx above),
        so a prior call's .bin files must survive. Stale .bin files from
        args removed between saves are not cleaned up; regenerate into a
        fresh directory to avoid drift.
        """
        buf_dir = _buf_dir_for(path)
        with open(path, "w") as f:
            f.write(self.to_json(buf_dir, kernel_idx=kernel_idx))


def _referenced_bin_files(kernels: list) -> set:
    names = set()
    for k in kernels:
        for a in k["args"]:
            init = a.get("init")
            if init:
                names.add(Path(init).name)
    return names


def save_launches(launches: list, path: str) -> None:
    """Serialize a LAUNCHES list to a multi-kernel JSON file.

    Buffer address assignment is deferred to the host driver. This function only
    serializes the semantic descriptor (elf, grid, arg kinds, shared_ids).
    Buffer init data is written to .bin files under a BUF_INIT_DIRNAME
    sub-directory next to the JSON file; see KernelLaunch.to_dict.

    buf_dir is pruned only after the new kernels list is fully built, and
    the JSON is only overwritten after that -- so a failure partway through
    (bad arg, write error) leaves the previous JSON and its buf_init/ files
    intact instead of leaving the old JSON pointing at deleted .bin files.
    Stale .bin files from a previous generation (e.g. a since-removed buffer
    arg) are removed once the new set is known to be complete.
    """
    buf_dir = _buf_dir_for(path)
    shared_written = set()
    kernels = [launch.to_dict(buf_dir, kernel_idx=i, shared_written=shared_written)
               for i, launch in enumerate(launches)]

    if buf_dir.is_dir():
        keep = _referenced_bin_files(kernels)
        for f in buf_dir.iterdir():
            if f.name not in keep:
                f.unlink()

    with open(path, "w") as f:
        json.dump({"kernels": kernels}, f, indent=2)
