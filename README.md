# Kernel Launch Descriptor

In a production OpenCL deployment, host code explicitly manages kernel buffer
allocations, directions, and inter-kernel dependencies. When running kernels on
a simulated device -- without an OpenCL runtime -- that host-side intent has no
place to live.

This library fills that gap. A `launch.py` file captures everything the host
would otherwise express: which kernels to launch, their grid dimensions, scalar
arguments, private buffer allocations, and shared-buffer dependencies across
kernels. `gen_launches_json.py` serializes this to a JSON file. The host driver
(TestDriver) reads the JSON via `parse_launches`, allocates memory, and sets up 
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
from kernel_launch import KernelLaunch, SharedBuffer, scalar, buffer, IN, OUT

result = SharedBuffer(256)

LAUNCHES = [
    KernelLaunch(
        elf="1CR/Producer/Producer.elf",
        grid=(1, 1, 4),
        args=[
            scalar("i32", 10),
            result.as_output(),
        ]
    ),
    KernelLaunch(
        elf="1CR/Consumer/Consumer.elf",
        grid=(2, 1, 4),
        args=[
            result.as_input(),
            buffer(128, OUT),
        ]
    ),
]
```

`LAUNCHES` must be a list, even for a single kernel.

## Python API

### `KernelLaunch(elf, grid, args)`

| Field  | Type                                           | Description                              |
|--------|------------------------------------------------|------------------------------------------|
| `elf`  | `str`                                          | Kernel ELF path                          |
| `grid` | `(n_x, n_y, n_ces)`                           | CR grid dimensions and CEs per CR        |
| `args` | `list[ScalarArg\|BufferArg\|SharedBufferView]` | Kernel arguments, in declaration order   |

### `scalar(type, value)`

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

### `buffer(size, dir, init=None)`

Private buffer -- one physical allocation per kernel, not shared.

| Parameter | Type     | Description                                      |
|-----------|----------|--------------------------------------------------|
| `size`    | `int`    | Bytes; must be 4-byte aligned                    |
| `dir`     | `BufDir` | `IN`, `OUT`, or `INOUT`                          |
| `init`    | `bytes`  | Optional -- host writes this before kernel runs  |

### `SharedBuffer(size, init=None)`

One physical allocation shared across all kernels that reference it.

| Parameter | Type    | Description                                                     |
|-----------|---------|-----------------------------------------------------------------|
| `size`    | `int`   | Bytes; must be 4-byte aligned                                   |
| `init`    | `bytes` | Optional -- host writes once before any kernel launches         |

```python
lut = SharedBuffer(1024, init=bytes(range(256)))  # pre-populated read-only input
buf = SharedBuffer(256)                            # kernel writes first
```

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

```json
{
  "kernels": [
    {
      "elf":  "1CR/Producer/Producer.elf",
      "grid": [1, 1, 4],
      "args": [
        {"kind": "scalar", "type": "i32", "value": 10},
        {"kind": "shared_buffer", "shared_id": "sb_0", "size": 256, "dir": "out", "init": null}
      ]
    },
    {
      "elf":  "1CR/Consumer/Consumer.elf",
      "grid": [2, 1, 4],
      "args": [
        {"kind": "shared_buffer", "shared_id": "sb_0", "size": 256, "dir": "in", "init": null},
        {"kind": "buffer", "size": 128, "dir": "out", "init": null}
      ]
    }
  ]
}
```

Segment IDs are not in the JSON -- the host driver assigns them per kernel.

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
