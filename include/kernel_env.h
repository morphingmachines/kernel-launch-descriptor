#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

// ---------------------------------------------------------------------------
// Launch descriptor — C++ representation of launches.json
// ---------------------------------------------------------------------------

enum class Buf_dir { IN, OUT, INOUT };

struct Arg_scalar {
    std::string type;   // "i8","u8","i16","u16","i32","u32","f32"
    uint32_t    value;  // bit-cast to uint32 (little-endian, zero-extended)
};

struct Arg_buffer {
    uint32_t    size;       // bytes; multiple of 4
    Buf_dir     dir;
    std::string init_path;  // absolute path to raw init bytes; empty = no host init.
                             // The host driver mmaps this file and writes it directly
                             // to the device backing.
};

struct Arg_shared_buffer {
    std::string shared_id;  // ties same physical buffer across kernels
    uint32_t    size;       // bytes; multiple of 4
    Buf_dir     dir;
    std::string init_path;  // written once at first allocation; empty = no init.
                             // Same mmap-and-write convention as Arg_buffer.init_path.
};

using Kernel_arg = std::variant<Arg_scalar, Arg_buffer, Arg_shared_buffer>;

struct Kernel_desc {
    std::string             elf;   // absolute path to kernel ELF
    uint32_t                n_x;
    uint32_t                n_y;
    uint32_t                n_ces;
    std::vector<Kernel_arg> args;
};

struct Launch_desc {
    std::vector<Kernel_desc> kernels;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Parse launches JSON ({"kernels":[...]}) into a Launch_desc.
// Pure conversion — no side effects, no hardware interaction.
Launch_desc parse_launches(const char *json_path);
