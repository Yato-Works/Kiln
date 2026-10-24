cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED PICO_SDK_PATH)
    message(FATAL_ERROR "PICO_SDK_PATH must point to a pinned Pico SDK checkout")
endif()
if(NOT DEFINED KILN_GENERATED_DIR)
    message(FATAL_ERROR "KILN_GENERATED_DIR must contain model.c and budget_certificate.txt")
endif()

include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(kiln_rp2040 C CXX ASM)

pico_sdk_init()

set(KILN_GENERATED_DIR "${KILN_GENERATED_DIR}" CACHE PATH "kiln build output directory")
set(KILN_MODEL_C "${KILN_GENERATED_DIR}/model.c")
set(KILN_CERTIFICATE "${KILN_GENERATED_DIR}/budget_certificate.txt")

if(NOT EXISTS "${KILN_MODEL_C}" OR NOT EXISTS "${KILN_CERTIFICATE}")
    message(FATAL_ERROR
        "KILN_GENERATED_DIR must contain model.c and budget_certificate.txt: ${KILN_GENERATED_DIR}")
endif()

file(GLOB KILN_GENERATED_SOURCES CONFIGURE_DEPENDS "${KILN_GENERATED_DIR}/*.c")
list(FILTER KILN_GENERATED_SOURCES EXCLUDE REGEX "/main\\.c$")

file(STRINGS "${KILN_CERTIFICATE}" KILN_ARENA_LINE REGEX "Arena.*RAM")
if(NOT KILN_ARENA_LINE)
    message(FATAL_ERROR "could not find the audited arena line in ${KILN_CERTIFICATE}")
endif()
string(REGEX REPLACE ".*: *([0-9,]+) bytes.*" "\\1" KILN_ARENA_BYTES "${KILN_ARENA_LINE}")
string(REPLACE "," "" KILN_ARENA_BYTES "${KILN_ARENA_BYTES}")
if(NOT KILN_ARENA_BYTES MATCHES "^[0-9]+$")
    message(FATAL_ERROR "could not parse audited arena bytes from: ${KILN_ARENA_LINE}")
endif()

add_executable(kiln_rp2040
    main.c
    ${KILN_GENERATED_SOURCES}
)

target_compile_features(kiln_rp2040 PRIVATE c_std_11)
target_compile_definitions(kiln_rp2040 PRIVATE KILN_MODEL_ARENA_BYTES=${KILN_ARENA_BYTES})
target_include_directories(kiln_rp2040 PRIVATE "${KILN_GENERATED_DIR}")
target_link_libraries(kiln_rp2040 PRIVATE pico_stdlib hardware_timer)

# Keep the strict Kiln gate on the generated model and board harness only.
# Applying -pedantic -Werror to the SDK itself would turn unrelated, supported
# SDK extensions into failures and would not test Kiln's generated C.
set_source_files_properties(
    main.c
    ${KILN_GENERATED_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror;-pedantic"
)

# USB CDC is the default capture transport for the first reference board.
# UART can be enabled by changing this board adapter, without changing KIR or
# the generated model.
pico_enable_stdio_usb(kiln_rp2040 1)
pico_enable_stdio_uart(kiln_rp2040 0)
pico_add_extra_outputs(kiln_rp2040)

message(STATUS "Kiln audited arena: ${KILN_ARENA_BYTES} bytes")
message(STATUS "Kiln generated sources: ${KILN_GENERATED_SOURCES}")
