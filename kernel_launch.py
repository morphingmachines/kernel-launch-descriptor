from __future__ import annotations
import json
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# Sub-directory (next to the launches JSON) holding raw init-data blobs.
# When init is provided as bytes, the data is written here as a .bin file
# and the JSON holds a relative path. The C++ loader memory-maps the file
# and passes the pointer straight to the device memory interface -- no copy.
# When init is a str/Path naming an existing file, the JSON holds that file's
# absolute path (a relative init is resolved against the directory of the
# launch.py that declared it); nothing is written to BUF_INIT_DIRNAME.
BUF_INIT_DIRNAME = "buf_init"


def _buf_dir_for(json_path: str | Path) -> Path:
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


def _caller_dir() -> Path:
    """Directory of the source file that called into this module.

    buffer()/SharedBuffer() run while launch.py executes, so the first stack
    frame outside kernel_launch.py is the launch.py (or helper module) line
    that declared the buffer. Falls back to cwd when that code has no real
    file (REPL, exec of a string, ...).
    """
    here = Path(__file__).resolve()
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith("<") and Path(filename).resolve() != here:
            return Path(filename).resolve().parent
        frame = frame.f_back
    return Path.cwd()


def _resolve_init(init: Optional[bytes | str | Path]) -> Optional[bytes | Path]:
    """Resolve a str/Path init to an absolute Path.

    A relative path is taken relative to the directory of the source file that
    called buffer()/SharedBuffer() (normally launch.py) -- not the process cwd
    and not the launches JSON's directory. bytes/None pass through unchanged.
    """
    if isinstance(init, (str, Path)):
        p = Path(init)
        return (p if p.is_absolute() else _caller_dir() / p).resolve()
    return init


def _validate_buffer(label: str, size: int, init: Optional[bytes | Path]) -> None:
    """init must already be resolved (see _resolve_init)."""
    if size % 4 != 0:
        raise ValueError(f"{label} size {size} must be 4-byte aligned")
    if isinstance(init, (bytes, bytearray)):
        if len(init) > size:
            raise ValueError(f"{label} init length {len(init)} exceeds size {size}")
    elif isinstance(init, Path):
        if not init.is_file():
            raise ValueError(f"{label} init file not found: {init}")
        file_size = init.stat().st_size
        if file_size != size:
            raise ValueError(f"{label} init file {init} size {file_size} does not match buffer size {size}")


@dataclass
class BufferArg:
    size: int                  # bytes; must be 4-byte aligned
    dir:  BufDir
    init: Optional[bytes | str | Path] = None  # written to backing memory before kernel launch (IN/INOUT)
    name: Optional[str] = None          # kernel param name; used by gen_redefine_main.py
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

    def __init__(self, size: int, init: Optional[bytes | str | Path] = None):
        init = _resolve_init(init)
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


def buffer(size: int, dir: BufDir, init: Optional[bytes | str | Path] = None,
           name: Optional[str] = None) -> BufferArg:
    init = _resolve_init(init)
    _validate_buffer("Buffer", size, init)
    return BufferArg(size=size, dir=dir, init=init, name=name)


def _write_init(buf_dir: Path, data: bytes, filename: str) -> str:
    buf_dir.mkdir(parents=True, exist_ok=True)
    (buf_dir / filename).write_bytes(data)
    return str(buf_dir / filename)


def _buffer_init_filename(kernel_idx: int, arg_idx: int) -> str:
    return f"k{kernel_idx}_a{arg_idx}.bin"


def _shared_init_filename(shared_id: str) -> str:
    return f"{shared_id}.bin"


def _file_init_path(init: str | Path) -> str:
    """JSON path for a file init. buffer()/SharedBuffer() already resolved it;
    this only guards against a BufferArg built directly with a relative path,
    which has no launch.py directory to resolve against."""
    p = Path(init)
    if not p.is_absolute():
        raise ValueError(f"init file path must be absolute here (use buffer()/SharedBuffer() "
                         f"to resolve relative paths): {init}")
    return str(p)


def _encode_scalar(a: ScalarArg) -> dict:
    return {"kind": "scalar", "type": a.type, "value": a.value, "name": a.name}


def _encode_buffer(a: BufferArg, buf_dir: Path, kernel_idx: int, arg_idx: int) -> dict:
    init_path = None
    if isinstance(a.init, (bytes, bytearray)):
        init_path = _write_init(buf_dir, a.init, _buffer_init_filename(kernel_idx, arg_idx))
    elif isinstance(a.init, (str, Path)):
        init_path = _file_init_path(a.init)
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
    if isinstance(a.shared.init, (bytes, bytearray)):
        filename = _shared_init_filename(shared_id)
        if shared_id not in shared_written:
            _write_init(buf_dir, a.shared.init, filename)
            shared_written.add(shared_id)
        init_path = str(buf_dir / filename)
    elif isinstance(a.shared.init, (str, Path)):
        init_path = _file_init_path(a.shared.init)
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

        Buffer init handling depends on the init type:
        - bytes: written as a .bin file under buf_dir; JSON holds its path
          (absolute when buf_dir is, as save_launches() ensures).
        - str/Path: JSON holds the file's absolute path (resolved by
          buffer()/SharedBuffer()); no file is written.
        shared_written tracks which SharedBuffer ids already had their .bin
        written, so a bytes-init shared buffer is written at most once across
        multiple kernels in one save_launches() call.
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

    def save(self, path: str | Path, kernel_idx: int = 0) -> None:
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
    """Resolved paths of every init file the encoded kernels reference --
    generated bytes blobs and user-provided files alike -- so pruning never
    removes a file the new JSON points at."""
    return {Path(a["init"]).resolve() for k in kernels for a in k["args"] if a.get("init")}


def _check_no_init_clobber(launches: list, buf_dir: Path) -> None:
    """Refuse to write a bytes-init blob over a user-provided init file.

    A str/Path init may legitimately live inside buf_dir; if its name matches
    a generated blob name (k<i>_a<j>.bin / <shared_id>.bin) the save would
    silently overwrite the user's data.
    """
    user_files = set()
    generated = {}
    for ki, launch in enumerate(launches):
        for ai, a in enumerate(launch.args):
            if isinstance(a, BufferArg):
                init, name = a.init, _buffer_init_filename(ki, ai)
            elif isinstance(a, SharedBufferView):
                init, name = a.shared.init, _shared_init_filename(a.shared._shared_id)
            else:
                continue
            if isinstance(init, (bytes, bytearray)):
                generated[buf_dir / name] = f"kernel {ki} arg {ai}"
            elif isinstance(init, (str, Path)):
                user_files.add(Path(init).resolve())
    for path, where in generated.items():
        if path in user_files:
            raise ValueError(f"bytes init for {where} would overwrite user init file {path}; "
                             f"move or rename that file out of {buf_dir}")


def save_launches(launches: list, path: str | Path, buf_dir: Optional[Path] = None) -> None:
    """Serialize a LAUNCHES list to a multi-kernel JSON file.

    Buffer address assignment is deferred to the host driver. This function only
    serializes the semantic descriptor (elf, grid, arg kinds, shared_ids).
    bytes init data is written as .bin files under buf_dir (default: BUF_INIT_DIRNAME
    sub-directory next to the JSON file). Pass buf_dir explicitly to place the
    buf_init/ directory elsewhere (e.g. next to launch.py). str/Path init is
    referenced in the JSON by absolute path (relative paths were resolved
    against the declaring launch.py's directory by buffer()/SharedBuffer());
    those files are not copied, and are never pruned even if they sit in
    buf_dir. Only regular files directly in buf_dir are pruned.

    buf_dir is pruned only after the new kernels list is fully built, and
    the JSON is only overwritten after that -- so a failure partway through
    (bad arg, write error) leaves the previous JSON and its buf_init/ files
    intact instead of leaving the old JSON pointing at deleted .bin files.
    Stale .bin files from a previous generation (e.g. a since-removed buffer
    arg) are removed once the new set is known to be complete.
    """
    if buf_dir is None:
        buf_dir = _buf_dir_for(path)
    buf_dir = Path(buf_dir).resolve()
    _check_no_init_clobber(launches, buf_dir)
    shared_written = set()
    kernels = [launch.to_dict(buf_dir, kernel_idx=i, shared_written=shared_written)
               for i, launch in enumerate(launches)]

    if buf_dir.is_dir():
        keep = _referenced_bin_files(kernels)
        for f in buf_dir.iterdir():
            if f.is_file() and f.resolve() not in keep:
                f.unlink()

    with open(path, "w") as f:
        json.dump({"kernels": kernels}, f, indent=2)
