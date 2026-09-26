/* Fixed-point frontend glued to the generated classifier.
 *
 * The Kiln C99 backend emits a `kiln_predict(const uint8_t *, int)` entry point
 * for the 10x49 int8 log-mel graph.  This translation unit only forwards the
 * frontend's int8 spectrogram into that model, so a device can go straight from
 * raw 16 kHz PCM to a predicted class index.
 *
 * Link order: the generated model source, kws_frontend.c, kws_pipeline.c.
 */

#include "kws_pipeline.h"

#include "kws_frontend.h"

int8_t kws_predict_raw_pcm(const int16_t *pcm_samples, size_t num_samples) {
    int8_t features[KWS_N_MELS * KWS_N_FRAMES];
    if (pcm_samples == NULL) {
        return -1;
    }
    kws_frontend_process(pcm_samples, num_samples, features);
    return kiln_predict((const uint8_t *)features, KWS_N_MELS * KWS_N_FRAMES);
}
