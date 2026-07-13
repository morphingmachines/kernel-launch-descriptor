# kernel-launch-descriptor: parses JSON launch descriptors and executes kernel
# setup (MMU segment allocation, DRAM layout, arg marshalling).
#
# When built standalone, set KLD_PLATFORM_INCLUDE_DIR to the directory
# containing mmu.h, mmu_segment_manager.h, and host_ctrl_ifc.hpp, and
# provide the fesvr/TileLink include paths via KLD_EXTRA_INCLUDE_DIRS.
#
# When consumed via add_subdirectory (normal use), the parent project
# sets these on the kernel_launch_descriptor target after this file runs.

cmake_minimum_required(VERSION 3.14)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(kernel_launch_descriptor CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_library(kernel_launch_descriptor STATIC kernel_env.cc)

target_include_directories(kernel_launch_descriptor PUBLIC include)

# nlohmann/json must be provided by the parent (or a find_package call
# in standalone mode before this file is included).
target_link_libraries(kernel_launch_descriptor PUBLIC nlohmann_json::nlohmann_json)
