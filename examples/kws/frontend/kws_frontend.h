#ifndef KWS_FRONTEND_H
#define KWS_FRONTEND_H

#include <stdint.h>
#include <stddef.h>
#include "kws_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Computes 10x49 log-mel spectrogram from raw 16kHz 16-bit PCM samples.
 * Output is row-major: out_spectrogram[m * 49 + frame] in signed int8 [-127, 127]
 * strictly adhering to KIR.md Section 2.2 and Section 2.3.
 *
 * Guaranteed zero-malloc, static memory only.
 */
void kws_frontend_process(
    const int16_t *pcm_samples,
    size_t num_samples,
    int8_t out_spectrogram[KWS_N_MELS * KWS_N_FRAMES]
);

#ifdef __cplusplus
}
#endif

#endif /* KWS_FRONTEND_H */
