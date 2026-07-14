#pragma once

#include <cstdint>

// Mirrors __kernel_exec_cmd from pocl_device.h (CwithHyperOps/include/pocl/pocl_device.h).
// ALIGN8(uint field) expands to: uint32_t field __attribute__((aligned(8))), giving
// 4 bytes value + 4 bytes implicit padding per field; 6 fields * 8 = 48 bytes total.
// TestDriver populates only args (logical address of the args array) and args_size
// (byte count). All other fields are written as zero.
struct Kernel_exec_cmd {
    uint32_t kernel_meta; uint32_t pad0;
    uint32_t args;        uint32_t pad1;
    uint32_t args_size;   uint32_t pad2;
    uint32_t ctx;         uint32_t pad3;
    uint32_t ctx_size;    uint32_t pad4;
    uint32_t status;      uint32_t pad5;
};
static_assert(sizeof(Kernel_exec_cmd) == 48,
              "Kernel_exec_cmd layout mismatch with pocl_device.h __kernel_exec_cmd");
