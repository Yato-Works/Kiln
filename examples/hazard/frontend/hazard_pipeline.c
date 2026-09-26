/* Fixed-point tokenizer glued to the generated classifier.
 *
 * The Kiln C99 backend emits a `kiln_predict(const uint8_t *, int)` entry point
 * for the 96-token uint8 bag graph.  This translation unit forwards the
 * tokenizer's uint8 token vector into that model, so a runtime can go straight
 * from a raw code snippet to a predicted hazard class.
 *
 * Link order: the generated model source, hazard_tokenizer.c, hazard_pipeline.c.
 */

#include "hazard_pipeline.h"

int8_t hazard_predict_source(const uint8_t *source, size_t length) {
    uint8_t tokens[HAZARD_TOKEN_COUNT];
    if (source == NULL && length > 0) {
        return -1;
    }
    hazard_tokenize(source, length, tokens);
    return kiln_predict(tokens, HAZARD_TOKEN_COUNT);
}
