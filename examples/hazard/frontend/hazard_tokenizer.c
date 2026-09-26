#include "hazard_tokenizer.h"

#define HAZARD_SPACE ((uint8_t)0x20)
#define HAZARD_QUESTION ((uint8_t)0x3F)
#define HAZARD_SENTINEL ((uint8_t)0x20)

uint32_t hazard_fnv1a32(const uint8_t *data, size_t length) {
    uint32_t value = 0x811C9DC5u;
    size_t index;

    for (index = 0; index < length; ++index) {
        value ^= (uint32_t)data[index];
        value *= 0x01000193u;
    }

    return value;
}

/*
 * The contract's only lossy rule.  Bytes >= 0x80 cannot be addressed by the
 * graph's uint8 EMBEDDING indices, so they fold to '?'; the corpus manifest
 * asserts that training and evaluation snippets are ASCII, which is why this is
 * a documented policy rather than a silent corruption.
 */
static uint8_t _normalize_byte(uint8_t byte) {
    if (byte >= (uint8_t)'A' && byte <= (uint8_t)'Z') {
        return (uint8_t)(byte + 0x20u);
    }
    if (byte < HAZARD_SPACE || byte == (uint8_t)0x7Fu) {
        return HAZARD_SPACE;
    }
    if (byte >= (uint8_t)0x80u) {
        return HAZARD_QUESTION;
    }

    return byte;
}

void hazard_tokenize(
    const uint8_t *source,
    size_t length,
    uint8_t out_tokens[HAZARD_TOKEN_COUNT]
) {
    uint8_t normal[HAZARD_WINDOW_BYTES + 1];
    size_t keep = length < (size_t)HAZARD_WINDOW_BYTES ? length : (size_t)HAZARD_WINDOW_BYTES;
    size_t index;

    for (index = 0; index < keep; ++index) {
        normal[index] = _normalize_byte(source[index]);
    }
    for (index = keep; index < (size_t)HAZARD_WINDOW_BYTES; ++index) {
        normal[index] = HAZARD_SPACE;
    }
    /* Every byte gets a successor, so the pair count stays exact. */
    normal[HAZARD_WINDOW_BYTES] = HAZARD_SENTINEL;

    for (index = 0; index < (size_t)HAZARD_WINDOW_BYTES; ++index) {
        out_tokens[index] = (uint8_t)(normal[index] - 0x1Fu);
    }
    for (index = 0; index < (size_t)HAZARD_WINDOW_BYTES; ++index) {
        uint32_t bucket = (uint32_t)HAZARD_BIGRAM_BASE
            + hazard_fnv1a32(&normal[index], 2u) % (uint32_t)HAZARD_BIGRAM_BUCKETS;
        out_tokens[HAZARD_WINDOW_BYTES + index] = (uint8_t)bucket;
    }
}
