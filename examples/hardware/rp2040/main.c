/* RP2040 reference-board harness for a generated Kiln model. */

#include <stdint.h>
#include <stdio.h>

#include "hardware/clocks.h"
#include "hardware/timer.h"
#include "pico/stdlib.h"

#ifndef KILN_MODEL_ARENA_BYTES
#error "CMake must provide KILN_MODEL_ARENA_BYTES from budget_certificate.txt"
#endif

int8_t kiln_predict(const uint8_t *input_bytes, int len);

#define KILN_INPUT_BYTES 96
#define KILN_VECTOR_COUNT 5
#define KILN_STACK_MARKER ((uint8_t)0xA5)

/* These symbols are supplied by the Pico SDK linker script. */
extern unsigned char __StackBottom;
extern unsigned char __StackTop;
extern unsigned char __flash_binary_end;

static uint8_t k_vectors[KILN_VECTOR_COUNT][KILN_INPUT_BYTES];

static void kiln_prepare_vectors(void) {
    for (int vector = 0; vector < KILN_VECTOR_COUNT; ++vector) {
        for (int index = 0; index < KILN_INPUT_BYTES; ++index) {
            uint8_t value = 0;
            if (vector == 1) {
                value = (index % 2 == 0) ? 0x00 : 0xFF;
            } else if (vector == 2) {
                value = 0xFF;
            } else if (vector == 3) {
                value = 0x80;
            } else if (vector == 4) {
                value = (uint8_t)((index * 7 + 1) & 0xFF);
            }
            k_vectors[vector][index] = value;
        }
    }
}

static uintptr_t kiln_stack_pointer(void) {
    register uintptr_t value __asm("sp");
    return value;
}

static void kiln_prime_stack(void) {
    uintptr_t stack_pointer = kiln_stack_pointer();
    uintptr_t guard = 128u;
    uintptr_t end = stack_pointer > guard ? stack_pointer - guard : stack_pointer;
    volatile uint8_t *cursor = &__StackBottom;

    while ((uintptr_t)cursor < end && cursor < &__StackTop) {
        *cursor++ = KILN_STACK_MARKER;
    }
}

static uint32_t kiln_stack_high_water_bytes(void) {
    const uint8_t *cursor = &__StackBottom;

    while (cursor < &__StackTop && *cursor == KILN_STACK_MARKER) {
        ++cursor;
    }
    return (uint32_t)(&__StackTop - cursor);
}

static uint32_t kiln_flash_bytes(void) {
    /* RP2040 XIP flash is mapped at 0x10000000. */
    return (uint32_t)((uintptr_t)&__flash_binary_end - 0x10000000u);
}

int main(void) {
    uint32_t max_cycles = 0;
    uint32_t frequency_hz;

    stdio_init_all();
    sleep_ms(1500);
    kiln_prepare_vectors();
    kiln_prime_stack();
    frequency_hz = clock_get_hz(clk_sys);

    for (int vector = 0; vector < KILN_VECTOR_COUNT; ++vector) {
        uint64_t start_us = time_us_64();
        int8_t prediction = kiln_predict(k_vectors[vector], KILN_INPUT_BYTES);
        uint64_t elapsed_us = time_us_64() - start_us;
        uint32_t cycles = (uint32_t)((elapsed_us * frequency_hz) / 1000000u);

        if (cycles > max_cycles) {
            max_cycles = cycles;
        }
        printf("%d %d\n", vector, (int)prediction);
    }

    uint32_t stack_bytes = kiln_stack_high_water_bytes();
    uint32_t peak_ram_bytes = (uint32_t)KILN_MODEL_ARENA_BYTES + stack_bytes;
    printf(
        "KILN_METRIC cycles=%lu peak_ram_bytes=%lu flash_bytes=%lu stack_bytes=%lu\n",
        (unsigned long)max_cycles,
        (unsigned long)peak_ram_bytes,
        (unsigned long)kiln_flash_bytes(),
        (unsigned long)stack_bytes
    );
    fflush(stdout);

    for (;;) {
        tight_loop_contents();
    }
}
