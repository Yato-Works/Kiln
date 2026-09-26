#ifndef HAZARD_TOKENIZER_H
#define HAZARD_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Phase 2 fixed-point feature contract, frozen by M2.0 (ROADMAP.md).
 *
 * examples/hazard/tokenizer.py is the normative implementation and
 * tests/test_hazard_frontend.py keeps this file bit-exact against it.  The rules:
 *
 *  * the snippet is normalised to ASCII -- A-Z become lowercase, control bytes
 *    and 0x7F become spaces, bytes >= 0x80 become '?';
 *  * the normal form is truncated and space-padded to HAZARD_WINDOW_BYTES;
 *  * HAZARD_TOKEN_COUNT ids come out: HAZARD_WINDOW_BYTES byte ids
 *    (1..96, i.e. byte - 0x1F) followed by HAZARD_WINDOW_BYTES bigram ids
 *    (97..255, i.e. 97 + FNV-1a(pair) % 159) over each byte and a space
 *    sentinel, so the graph's single input tensor carries both channels;
 *  * id 0 is never emitted: padding happens in the byte domain, so a padded
 *    position is a real space token and a max-pool bag needs no special case.
 *
 * Pure function, no dynamic allocation, no floating point, no global state.
 */
#define HAZARD_WINDOW_BYTES 48
#define HAZARD_TOKEN_COUNT (2 * HAZARD_WINDOW_BYTES)
#define HAZARD_VOCAB_SIZE 256
#define HAZARD_CHARACTER_BASE 1
#define HAZARD_CHARACTER_COUNT 96
#define HAZARD_BIGRAM_BASE (HAZARD_CHARACTER_BASE + HAZARD_CHARACTER_COUNT)
#define HAZARD_BIGRAM_BUCKETS (HAZARD_VOCAB_SIZE - HAZARD_BIGRAM_BASE)

void hazard_tokenize(
    const uint8_t *source,
    size_t length,
    uint8_t out_tokens[HAZARD_TOKEN_COUNT]
);

/*
 * Deterministic FNV-1a (32-bit), the hash fastText uses for its subword
 * buckets.  Exposed so the tests can check the bigram channel directly.
 */
uint32_t hazard_fnv1a32(const uint8_t *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* HAZARD_TOKENIZER_H */
