# Generate from the build root (cwd) into build/fileinit/ (JSON dir), neither
# of which is this source directory -- relative init paths must still resolve
# to the .bin files next to launch.py.
add_launch_example(fileinit
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/p.bin" "${CMAKE_CURRENT_SOURCE_DIR}/lut.bin")
