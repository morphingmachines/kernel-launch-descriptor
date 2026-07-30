// Smoke-tests the Python-generate -> JSON -> C++-parse pipeline for one
// launches.json: parses it with parse_launches, prints a summary, and
// checks that every buffer/shared_buffer init_path actually opens and its
// content fits within the declared buffer size.
//
// This does not run the kernel or touch any simulated device -- it only
// validates that a launch.py descriptor produces a launches.json (and
// buf_init/ files) that the C++ side can consume correctly.
//
// Usage: check_launch <launches.json>
// Exit 0 on success, 1 on any parse or validation failure.
#include "kernel_env.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

bool check_init_path(const std::string &label, const std::string &init_path, uint32_t buf_size) {
    if (init_path.empty()) {
        printf("    %s: no init\n", label.c_str());
        return true;
    }
    std::ifstream f(init_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "    %s: FAILED to open init file '%s'\n", label.c_str(), init_path.c_str());
        return false;
    }
    const std::streamsize n = f.tellg();
    if (n < 0 || static_cast<uint32_t>(n) > buf_size) {
        fprintf(stderr, "    %s: init file '%s' size %lld exceeds buffer size %u\n", label.c_str(),
                init_path.c_str(), static_cast<long long>(n), buf_size);
        return false;
    }
    printf("    %s: init '%s' (%lld bytes) OK\n", label.c_str(), init_path.c_str(),
           static_cast<long long>(n));
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <launches.json>\n", argv[0]);
        return 1;
    }

    Launch_desc desc;
    try {
        desc = parse_launches(argv[1]);
    } catch (const std::exception &e) {
        fprintf(stderr, "parse_launches failed: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    for (size_t ki = 0; ki < desc.kernels.size(); ++ki) {
        const Kernel_desc &kd = desc.kernels[ki];
        printf("kernel[%zu] elf=%s grid=(%u,%u,%u) args=%zu\n", ki, kd.elf.c_str(), kd.n_x, kd.n_y,
               kd.n_ces, kd.args.size());
        if (kd.elf.empty()) {
            fprintf(stderr, "  kernel[%zu]: empty elf path\n", ki);
            ok = false;
        }

        for (size_t ai = 0; ai < kd.args.size(); ++ai) {
            const std::string label = "arg[" + std::to_string(ai) + "]";
            const Kernel_arg &arg = kd.args[ai];
            if (std::holds_alternative<Arg_scalar>(arg)) {
                const Arg_scalar &as = std::get<Arg_scalar>(arg);
                printf("    %s: scalar type=%s value=%u\n", label.c_str(), as.type.c_str(), as.value);
            } else if (std::holds_alternative<Arg_buffer>(arg)) {
                const Arg_buffer &ab = std::get<Arg_buffer>(arg);
                printf("    %s: buffer size=%u\n", label.c_str(), ab.size);
                ok = check_init_path(label, ab.init_path, ab.size) && ok;
            } else if (std::holds_alternative<Arg_shared_buffer>(arg)) {
                const Arg_shared_buffer &asb = std::get<Arg_shared_buffer>(arg);
                printf("    %s: shared_buffer id=%s size=%u\n", label.c_str(), asb.shared_id.c_str(),
                       asb.size);
                ok = check_init_path(label, asb.init_path, asb.size) && ok;
            }
        }
    }

    if (!ok) {
        fprintf(stderr, "FAIL\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
