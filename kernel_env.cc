#include "kernel_env.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t scalar_pack_u32(const nlohmann::json &a) {
    const std::string type = a.at("type");
    if (type == "i32" || type == "i16" || type == "i8")
        return static_cast<uint32_t>(a.at("value").get<int32_t>());
    if (type == "u32" || type == "u16" || type == "u8")
        return a.at("value").get<uint32_t>();
    if (type == "f32") {
        float fv = a.at("value").get<float>();
        uint32_t u;
        std::memcpy(&u, &fv, sizeof(fv));
        return u;
    }
    throw std::runtime_error("Unsupported scalar type: " + type);
}

static Buf_dir parse_dir(const std::string &s) {
    if (s == "in")
        return Buf_dir::IN;
    if (s == "out")
        return Buf_dir::OUT;
    if (s == "inout")
        return Buf_dir::INOUT;
    throw std::runtime_error("Unknown buffer direction: " + s);
}

// kernel_launch.py always writes absolute init paths (bytes blobs under the
// resolved buf_dir; file inits resolved against the declaring launch.py's
// directory), and std::filesystem's `/` returns an absolute right-hand side
// unchanged. The json_dir join only matters for hand-written JSON with a
// relative path, which is taken relative to the JSON's own directory.
static std::string resolve_init_path(const nlohmann::json &ja, const std::filesystem::path &json_dir) {
    if (!ja.contains("init") || ja.at("init").is_null())
        return "";
    return (json_dir / ja.at("init").get<std::string>()).string();
}

Launch_desc parse_launches(const char *json_path) {
    nlohmann::json j;
    {
        std::ifstream f(json_path);
        if (!f.is_open())
            throw std::runtime_error(std::string("Cannot open launches JSON: ") + json_path);
        f >> j;
    }
    const std::filesystem::path json_dir = std::filesystem::path(json_path).parent_path();

    Launch_desc desc;
    for (const auto &jk : j.at("kernels")) {
        Kernel_desc kd;
        kd.elf = jk.at("elf").get<std::string>();
        const auto &jgrid = jk.at("grid");
        kd.n_x = jgrid.at(0).get<uint32_t>();
        kd.n_y = jgrid.at(1).get<uint32_t>();
        kd.n_ces = jgrid.at(2).get<uint32_t>();

        for (const auto &ja : jk.at("args")) {
            const std::string kind = ja.at("kind").get<std::string>();

            if (kind == "scalar") {
                kd.args.push_back(Arg_scalar{
                    ja.at("type").get<std::string>(),
                    scalar_pack_u32(ja),
                });
            } else if (kind == "buffer") {
                Arg_buffer ab;
                ab.size = ja.at("size").get<uint32_t>();
                ab.dir = parse_dir(ja.at("dir").get<std::string>());
                ab.init_path = resolve_init_path(ja, json_dir);
                kd.args.push_back(std::move(ab));
            } else if (kind == "shared_buffer") {
                Arg_shared_buffer asb;
                asb.shared_id = ja.at("shared_id").get<std::string>();
                asb.size = ja.at("size").get<uint32_t>();
                asb.dir = parse_dir(ja.at("dir").get<std::string>());
                asb.init_path = resolve_init_path(ja, json_dir);
                kd.args.push_back(std::move(asb));
            } else {
                throw std::runtime_error("Unknown arg kind: " + kind);
            }
        }
        desc.kernels.push_back(std::move(kd));
    }
    return desc;
}
