# Standalone example project for kernel-launch-descriptor.
#
# Each subdirectory (fib/, bmm/) has a launch.py. Building this project
# generates that launch.py's launches.json (+ buf_init/ files) via
# gen_launches_json.py, then registers a ctest that parses the result with
# check_launch and validates it -- no RTL simulation involved.
#
# Usage:
#   cmake -S . -B build
#   cmake --build build
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.14)
project(kernel_launch_descriptor_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# Captured here (not read inside add_launch_example) because
# CMAKE_CURRENT_SOURCE_DIR tracks the directory of whichever CMakeLists.txt
# is currently executing -- inside the function that would be the calling
# example's own subdirectory (e.g. fib/), not this one.
set(EXAMPLES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DESCRIPTOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_subdirectory(${DESCRIPTOR_ROOT} kernel_launch_descriptor_build)

enable_testing()

add_executable(check_launch common/check_launch.cc)
target_link_libraries(check_launch PRIVATE kernel_launch_descriptor)

# add_launch_example(name): generates <name>/launches.json from <name>/launch.py
# and registers a ctest that validates it with check_launch.
function(add_launch_example name)
    set(launch_py "${EXAMPLES_ROOT}/${name}/launch.py")
    set(out_json  "${CMAKE_CURRENT_BINARY_DIR}/launches.json")
    add_custom_command(
        OUTPUT "${out_json}"
        COMMAND ${Python3_EXECUTABLE}
                "${DESCRIPTOR_ROOT}/gen_launches_json.py"
                "${launch_py}" -o "${out_json}"
        DEPENDS "${launch_py}" "${DESCRIPTOR_ROOT}/kernel_launch.py"
        COMMENT "Generating ${name}/launches.json"
    )
    add_custom_target(${name}_launches ALL DEPENDS "${out_json}")
    add_test(NAME ${name}_check COMMAND check_launch "${out_json}")
endfunction()

add_subdirectory(fib)
add_subdirectory(bmm)
add_subdirectory(pipeline)
