# Kernel Launch Descriptor

In a production OpenCL deployment, host code explicitly manages kernel buffer
allocations, directions, and inter-kernel dependencies. When running kernels on
a simulated device -- without an OpenCL runtime -- that host-side intent has no
place to live.

This library fills that gap. A `launch.py` file captures everything the host
would otherwise express: which kernels to launch, their grid dimensions, scalar
arguments, private buffer allocations, and shared-buffer dependencies across
kernels. `gen_launches_json.py` serializes this to a JSON file. The host driver
reads the JSON via `parse_launches`, allocates memory, and sets up 
each kernel's execution environment before invoking the kernel execution on the 
simulated device.

The descriptor layer is platform-agnostic -- it knows nothing about the simulated 
device. Those concerns belong to the host driver that consumes it.

## Generating JSON

```sh
./gen_launches_json.py launch.py              # writes launches.json next to launch.py
./gen_launches_json.py launch.py -o out.json  # explicit output path
```

See [example_launch.py](example_launch.py) for a complete working example.

## Writing a launch.py

```python
from pathlib import Path
from kernel_launch import KernelLaunch, SharedBuffer, scalar, buffer, IN, OUT

# Absolute path to the mm-baremetal-examples directory.
MM_EXAMPLES = Path("/path/to/mm-baremetal-examples")

result = SharedBuffer(256)

LAUNCHES = [
    KernelLaunch(
        elf=str(MM_EXAMPLES / "1CR/Producer/Producer.elf"),
        entry="producerStart",
        grid=(1, 1, 4),
        args=[
            scalar("i32", 10, name="n"),
            result.as_output(),
        ]
    ),
    KernelLaunch(
        elf=str(MM_EXAMPLES / "1CR/Consumer/Consumer.elf"),
        entry="consumerStart",
        grid=(2, 1, 4),
        args=[
            result.as_input(),
            buffer(128, OUT, name="out"),
        ]
    ),
]
```

`LAUNCHES` must be a list, even for a single kernel.

## Python API

### `KernelLaunch(elf, grid, args)`

| Field   | Type                                           | Description                                              |
|---------|------------------------------------------------|----------------------------------------------------------|
| `elf`   | `str`                                          | Absolute path to kernel ELF                              |
| `entry` | `str`                                          | Kernel entry function name (e.g. `"producerStart"`)      |
| `grid`  | `(n_x, n_y, n_ces)`                           | CR grid dimensions and CEs per CR                        |
| `args`  | `list[ScalarArg\|BufferArg\|SharedBufferView]` | Kernel arguments, in declaration order                   |

### `scalar(type, value)`

`scalar(type, value, name=None)` -- `name` is the kernel param name, used by `gen_redefine_main.py`.

Packed as `uint32` in the args slot. Supported types:

| Type  | C equivalent | Width   |
|-------|--------------|---------|
| `i8`  | `int8_t`     | 1 byte  |
| `u8`  | `uint8_t`    | 1 byte  |
| `i16` | `int16_t`    | 2 bytes |
| `u16` | `uint16_t`   | 2 bytes |
| `i32` | `int32_t`    | 4 bytes |
| `u32` | `uint32_t`   | 4 bytes |
| `f32` | `float`      | 4 bytes |

Values narrower than 32 bits are zero-extended to fill the slot.

### `buffer(size, dir, init=None, name=None)`

Private buffer -- one physical allocation per kernel, not shared.

| Parameter | Type           | Description                                                      |
|-----------|----------------|------------------------------------------------------------------|
| `size`    | `int`          | Bytes; must be 4-byte aligned                                    |
| `dir`     | `BufDir`       | `IN`, `OUT`, or `INOUT`                                          |
| `init`    | `bytes \| str \| Path` | Optional -- host writes this before kernel runs. `bytes`: written to `buf_init/` on save. `str`/`Path`: existing file referenced directly; must match `size` exactly. See [File init paths](#file-init-paths). |
| `name`    | `str`          | Optional -- kernel param name; used by `gen_redefine_main.py`   |

### `SharedBuffer(size, init=None)`

One physical allocation shared across all kernels that reference it.

| Parameter | Type           | Description                                                     |
|-----------|----------------|-----------------------------------------------------------------|
| `size`    | `int`          | Bytes; must be 4-byte aligned                                   |
| `init`    | `bytes \| str \| Path` | Optional -- host writes once before any kernel launches. `bytes`: written to `buf_init/` on save. `str`/`Path`: existing file referenced directly; must match `size` exactly. See [File init paths](#file-init-paths). |

```python
lut = SharedBuffer(1024, init=bytes(range(256)))  # pre-populated read-only input
buf = SharedBuffer(256)                            # kernel writes first
```

### File init paths

A relative `str`/`Path` `init` is resolved against the **directory of the
`launch.py` that declares it**. It is not resolved against the process cwd or
the directory the launches JSON is written to. Resolution and validation
(file exists, size == buffer `size`) happen right away inside
`buffer()`/`SharedBuffer()`, so a bad path raises `ValueError` at the offending
line, and the message names the resolved absolute path. The JSON always holds
that absolute path.

```python
buffer(64, IN, init="p.bin")            # -> <launch.py dir>/p.bin
buffer(64, IN, init="data/p.bin")       # -> <launch.py dir>/data/p.bin
buffer(64, IN, init="/abs/p.bin")       # absolute: used as-is
```

Strictly, "the declaring file" is the source file of the code that called
`buffer()`/`SharedBuffer()`. If a helper module in another directory builds
buffers for `launch.py`, its relative paths resolve against the helper's
directory. Code with no source file (REPL, `exec` of a string) resolves
against the cwd. See [examples/fileinit/](examples/fileinit/).

Produce a `SharedBufferView` for an arg slot:

| Method            | Direction | Meaning                           |
|-------------------|-----------|-----------------------------------|
| `buf.as_input()`  | `IN`      | Kernel reads                      |
| `buf.as_output()` | `OUT`     | Kernel writes                     |
| `buf.as_inout()`  | `INOUT`   | Kernel reads and writes           |

### `save_launches(launches, path)`

Serializes a `LAUNCHES` list to a JSON file.

```python
from kernel_launch import save_launches
save_launches(launches, "launches.json")
```

## JSON format

Output of `gen_launches_json.py` / `save_launches`. Input to `parse_launches`.

Buffer `init` data is serialized as a file path string or `null` in the JSON.
Two sources are supported:

- **`bytes` init**: written to a `buf_init/` sub-directory next to `launch.py`
  as a `.bin` blob. `init` in the JSON holds the absolute path to that blob.
  Filename conventions:
  - Private buffer: `buf_init/k<kernel_index>_a<arg_index>.bin`
  - Shared buffer: `buf_init/<shared_id>.bin` (e.g. `buf_init/sb_0.bin`)

  `save_launches()` manages this directory -- stale blobs from removed args
  are pruned on each save. Files referenced by a `str`/`Path` init are never
  pruned, even if they sit inside `buf_init/`, and subdirectories are left
  alone. A save that would write a `bytes` blob over a user-provided init
  file of the same name raises `ValueError`.
- **`str`/`Path` init**: an existing `.bin` file on the host. `init` in the JSON
  holds its absolute path (relative paths are resolved against the declaring
  `launch.py`'s directory -- see [File init paths](#file-init-paths)). The file
  is not copied; it must remain accessible at that path when the host driver
  loads the JSON.

So every `init` in generated JSON is absolute, and the JSON can live in any
directory. `parse_launches` still joins a relative `init` (hand-written JSON
only) onto the JSON's own directory.

In both cases the C++ loader memory-maps the file and writes it straight to device
backing (see `Arg_buffer::init_path` / `Arg_shared_buffer::init_path` in
`include/kernel_env.h`).

```json
{
  "kernels": [
    {
      "elf":   "/abs/path/to/mm-baremetal-examples/1CR/Producer/Producer.elf",
      "entry": "producerStart",
      "grid":  [1, 1, 4],
      "args": [
        {"kind": "scalar", "type": "i32", "value": 10, "name": "n"},
        {"kind": "shared_buffer", "shared_id": "sb_0", "size": 256, "dir": "out", "init": null}
      ]
    },
    {
      "elf":   "/abs/path/to/mm-baremetal-examples/1CR/Consumer/Consumer.elf",
      "entry": "consumerStart",
      "grid":  [2, 1, 4],
      "args": [
        {"kind": "shared_buffer", "shared_id": "sb_0", "size": 256, "dir": "in", "init": "/abs/path/to/launch_dir/buf_init/sb_0.bin"},
        {"kind": "buffer", "size": 128, "dir": "out", "init": null, "name": "out"}
      ]
    }
  ]
}
```

Buffer addresses are not in the JSON -- the host driver assigns them per kernel.

## C++ API

```cpp
#include "kernel_env.h"

Launch_desc desc = parse_launches("launches.json");
for (const Kernel_desc &k : desc.kernels) {
    // k.elf, k.n_x, k.n_y, k.n_ces, k.args
}
```

`parse_launches` returns one `Launch_desc` containing all kernels. Each
`Kernel_desc` holds `elf`, `n_x`, `n_y`, `n_ces`, and `args` -- a
`std::vector<Kernel_arg>` where each element is a
`std::variant<Arg_scalar, Arg_buffer, Arg_shared_buffer>`.

### CMake

```cmake
add_subdirectory(kernel-launch-descriptor)
target_link_libraries(MyDriver PRIVATE kernel_launch_descriptor)
```

`kernel_launch_descriptor` pulls in `nlohmann_json` automatically.

## Examples

[examples/](examples/) has four `launch.py` descriptors:

| Example                                             | Kernels | Demonstrates                                                        |
|------------------------------------------------------|---------|-----------------------------------------------------------------------|
| [examples/fib/launch.py](examples/fib/launch.py)     | 1       | Mirrors `mm-baremetal-examples/1CR/Fib`, points at its real `.elf`   |
| [examples/bmm/launch.py](examples/bmm/launch.py)     | 1       | Mirrors `.../1CR/BMM`; exercises the buffer `init` path (see JSON format above) |
| [examples/pipeline/launch.py](examples/pipeline/launch.py) | 3 | Synthetic Producer -> Filter -> Consumer; two `SharedBuffer`s chaining data across kernels (ELFs are illustrative, not real) |
| [examples/fileinit/launch.py](examples/fileinit/launch.py) | 1 | Relative file `init` (`init="p.bin"`) on a `buffer` and a `SharedBuffer`; generated from a different cwd into a third directory, JSON still holds absolute paths to the files next to `launch.py` |

Building generates each `launch.py`'s `launches.json` (+ `buf_init/`) via
`gen_launches_json.py`, then runs [examples/common/check_launch.cc](examples/common/check_launch.cc)
against it as a ctest -- parses the JSON with `parse_launches` and confirms
every buffer's `init` file opens and fits its declared size. The
`kernel_launch_py_tests` ctest runs the Python unit tests in [tests/](tests/)
(init path resolution, error messages, `buf_init/` pruning). No RTL
simulation involved.

```sh
cmake -S examples -B examples/build
cmake --build examples/build
ctest --test-dir examples/build --output-on-failure
```
