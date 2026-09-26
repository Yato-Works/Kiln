#include "kws_frontend.h"
#include <limits.h>

/*
 * torch.stft(center=True, n_fft=1024, hop=320) yields
 * 1 + (16000 + 2*(1024/2) - 1024)/320 = 51 frames.  The reference normalizes the
 * log-mel spectrogram by the peak over *all* of them and only then crops to the
 * 49 frames the classifier consumes, so the frontend must analyse 51 frames too.
 */
#define KWS_ANALYSIS_FRAMES (KWS_N_FRAMES + 2)

/*
 * The Hann window is stored in Q15.  Windowing keeps six extra fractional bits
 * ((sample * w_q15) >> 9 instead of >> 15) so the per-sample rounding error is
 * 2^-6 LSB instead of 1/2 LSB; that keeps the fixed-point noise floor about
 * 36 dB below the FP32 reference even in near-silent frames.  The FFT is then
 * scaled up by 2^6, which is undone exactly when the power spectrum is formed
 * (KWS_POWER_SHIFT = 2 * (15 - KWS_WINDOW_SHIFT)).
 *
 * KWS_WINDOW_SHIFT is calibrated, not arbitrary: at 15 (no extra bits) the
 * rounding error of the windowing multiplication lifts the noise floor of a
 * quiet bin to about one int8 step, which breaks the parity contract.
 */
#define KWS_WINDOW_SHIFT 9
#define KWS_WINDOW_ROUND (1 << (KWS_WINDOW_SHIFT - 1))
#define KWS_POWER_SHIFT (2 * (15 - KWS_WINDOW_SHIFT))
#define KWS_POWER_ROUND (1 << (KWS_POWER_SHIFT - 1))

/*
 * Static scratchpad memory for frame processing.
 * 100% Zero dynamic allocation.
 */
static int32_t _fft_re[KWS_FFT_SIZE];
static int32_t _fft_im[KWS_FFT_SIZE];
static uint64_t _power_spec[KWS_FFT_SIZE / 2 + 1];
static uint64_t _mel_energies[KWS_N_MELS * KWS_ANALYSIS_FRAMES];

/* Fixed-point log2 in Q16: log2(val) * 65536 */
static uint32_t _fixed_log2_q16(uint64_t val) {
    if (val == 0) return 0;
    
    // Find highest bit position
    int k = 0;
    uint64_t temp = val;
    while (temp >>= 1) {
        k++;
    }
    
    uint32_t frac_idx;
    if (k <= 6) {
        /* The six fraction bits are exact for these tiny mantissas. */
        frac_idx = (uint32_t)((val << (6 - k)) & 0x3Fu);
    } else {
        /* Round the six fraction bits after the leading one to nearest. */
        uint32_t top7 = (uint32_t)((val >> (k - 7)) & 0x7Fu);
        frac_idx = (top7 + 1u) >> 1;
        if (frac_idx == 64u) {
            /* Rounded up into the next power of two. */
            k += 1;
            frac_idx = 0u;
        }
    }
    uint32_t lut_val = (uint32_t)kws_log2_lut[frac_idx];
    return ((uint32_t)k << 16) + lut_val;
}

/* In-place radix-2 DIT FFT of 1024 points */
static void _fft_1024(int32_t *re, int32_t *im) {
    // 1. Bit reversal permutation
    for (int i = 0; i < KWS_FFT_SIZE; ++i) {
        int rev = (int)kws_bit_reverse[i];
        if (i < rev) {
            int32_t tr = re[i]; re[i] = re[rev]; re[rev] = tr;
            int32_t ti = im[i]; im[i] = im[rev]; im[rev] = ti;
        }
    }

    // 2. Cooley-Tukey butterfly stages (10 stages for 1024)
    for (int stage = 0; stage < 10; ++stage) {
        int m = 1 << stage;
        int step = m << 1;
        int twiddle_stride = 512 / m;

        for (int k = 0; k < m; ++k) {
            int tw_idx = k * twiddle_stride;
            int32_t tw_cos = (int32_t)kws_twiddle_cos[tw_idx];
            int32_t tw_sin = (int32_t)kws_twiddle_sin[tw_idx];

            for (int i = k; i < KWS_FFT_SIZE; i += step) {
                int j = i + m;
                // Complex multiplication with Q15 twiddle
                int64_t tr = (int64_t)re[j] * tw_cos - (int64_t)im[j] * tw_sin;
                int64_t ti = (int64_t)re[j] * tw_sin + (int64_t)im[j] * tw_cos;
                /* Q15 twiddle multiply, rounded to nearest instead of floored so
                 * quiet bins do not drift away from the FP32 reference. */
                int32_t t_re = (int32_t)((tr + 16384) >> 15);
                int32_t t_im = (int32_t)((ti + 16384) >> 15);

                re[j] = re[i] - t_re;
                im[j] = im[i] - t_im;
                re[i] = re[i] + t_re;
                im[i] = im[i] + t_im;
            }
        }
    }
}

/*
 * Match torch.stft(center=True, pad_mode="reflect") for the KWS reference
 * preprocessor.  The reference first pads/truncates the waveform to one
 * second, then reflects n_fft/2 samples around each edge (a single reflection
 * that does not repeat the edge sample): index -1 -> 1 and index N -> N - 2.
 * Samples past the caller's buffer but inside the one-second window read as the
 * zero padding the reference applies to short utterances.
 */
static int16_t _reflected_sample(
    const int16_t *pcm_samples,
    size_t num_samples,
    ptrdiff_t index
) {
    const ptrdiff_t last = (ptrdiff_t)KWS_SAMPLE_RATE - 1;
    if (index < 0) {
        index = -index;
    } else if (index > last) {
        index = 2 * last - index;
    }
    if (index < 0 || index >= (ptrdiff_t)num_samples) {
        return 0;
    }
    return pcm_samples[(size_t)index];
}

void kws_frontend_process(
    const int16_t *pcm_samples,
    size_t num_samples,
    int8_t out_spectrogram[KWS_N_MELS * KWS_N_FRAMES]
) {
    uint64_t max_mel_energy = 0;
    if (num_samples > KWS_SAMPLE_RATE) num_samples = KWS_SAMPLE_RATE;

    /* STFT center padding is n_fft/2 = 512 samples. */
    for (int frame = 0; frame < KWS_ANALYSIS_FRAMES; ++frame) {
        /* The 640-sample window is centered on each STFT hop. */
        ptrdiff_t start_idx =
            (ptrdiff_t)frame * KWS_HOP_LEN - (ptrdiff_t)(KWS_WIN_LEN / 2);

        /* Torch pads the 640-sample window to the 1024-point FFT. */
        const int window_pad = (KWS_FFT_SIZE - KWS_WIN_LEN) / 2;
        for (int i = 0; i < KWS_FFT_SIZE; ++i) {
            if (i >= window_pad && i < window_pad + KWS_WIN_LEN) {
                int window_i = i - window_pad;
                int16_t sample = _reflected_sample(
                    pcm_samples, num_samples, start_idx + window_i
                );
                int32_t win_sample = (int32_t)(
                    ((int64_t)sample * kws_hann_window[window_i]
                     + KWS_WINDOW_ROUND) >> KWS_WINDOW_SHIFT
                );
                _fft_re[i] = win_sample;
            } else {
                _fft_re[i] = 0;
            }
            _fft_im[i] = 0;
        }

        _fft_1024(_fft_re, _fft_im);

        for (int k = 0; k <= KWS_FFT_SIZE / 2; ++k) {
            int64_t r = _fft_re[k];
            int64_t im = _fft_im[k];
            /* Undo the 2^KWS_POWER_SHIFT window gain (rounded, not truncated). */
            _power_spec[k] = (uint64_t)(
                (r * r + im * im + KWS_POWER_ROUND) >> KWS_POWER_SHIFT
            );
        }

        int weight_offset = 0;
        for (int m = 0; m < KWS_N_MELS; ++m) {
            uint16_t start_bin = kws_mel_start_bin[m];
            uint16_t filter_len = kws_mel_filter_len[m];
            uint64_t acc = 0;
            for (int k = 0; k < filter_len; ++k) {
                uint64_t pwr = _power_spec[start_bin + k];
                int16_t w = kws_mel_weights[weight_offset + k];
                /* Q15 mel weight: scale the product, not the power alone. */
                acc += (pwr * (uint64_t)(uint16_t)w + 16384u) >> 15;
            }
            weight_offset += filter_len;
            _mel_energies[m * KWS_ANALYSIS_FRAMES + frame] = acc;
            if (acc > max_mel_energy) max_mel_energy = acc;
        }
    }

    /*
     * The Python reference normalizes each utterance by its maximum mel
     * energy, then clamps 10*log10 to [-80, 0] dB.  The centered int8 mapping
     * is exactly (dB + 40) / (40/127); it is evaluated as
     * 127 - drop_db / (40/127) so the division stays non-negative, because C
     * truncates toward zero and a negative numerator would bias every quiet bin
     * up by one step.
     *
     * Digital silence is a special case: the reference divides by
     * max(mel).clamp(min=1e-6) = 0, so ratio 0 -> -80 dB -> -127 everywhere.
     */
    uint32_t log2_max = _fixed_log2_q16(max_mel_energy);
    for (int m = 0; m < KWS_N_MELS; ++m) {
        for (int frame = 0; frame < KWS_N_FRAMES; ++frame) {
            uint64_t energy = _mel_energies[m * KWS_ANALYSIS_FRAMES + frame];
            int32_t int8_val = -127;
            if (energy > 0 && max_mel_energy > 0) {
                uint32_t log2_e = _fixed_log2_q16(energy);
                int64_t log2_diff = (int64_t)log2_max - (int64_t)log2_e;
                if (log2_diff < 0) log2_diff = 0;
                /* 10*log10(2) in Q16; keep dB in Q16. */
                int64_t drop_q16 = ((log2_diff * 197283LL) >> 16);
                int8_val = 127 - (int32_t)(
                    (drop_q16 * 127LL + (20LL << 16)) / (40LL << 16)
                );
            }
            if (int8_val < -127) int8_val = -127;
            if (int8_val > 127) int8_val = 127;
            out_spectrogram[m * KWS_N_FRAMES + frame] = (int8_t)int8_val;
        }
    }
}
