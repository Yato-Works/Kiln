import tarfile
import io
import time
import os
import torch
import torchaudio
from examples.kws.dataset import extract_logmel_spectrogram

label_map = {
    "_silence_": 0,
    "_unknown_": 1,
    "yes": 2,
    "no": 3,
    "up": 4,
    "down": 5,
    "left": 6,
    "right": 7,
    "on": 8,
    "off": 9,
    "stop": 10,
    "go": 11
}

cache_file = "examples/kws/kws_processed_data.pt"
checkpoint_file = "examples/kws/kws_precompute_ckpt.pt"

specs = []
labels = []
start_idx = 0

if os.path.exists(checkpoint_file):
    ckpt = torch.load(checkpoint_file)
    specs = ckpt["specs"]
    labels = ckpt["labels"]
    start_idx = len(specs)
    print(f"Resuming precomputation from index {start_idx}/4890...")
else:
    print("Starting precomputation from index 0/4890...")

step_start = time.time()
with tarfile.open("speech_commands_test_set_v0.02.tar.gz", "r:gz") as tar:
    wav_members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
    for idx in range(start_idx, len(wav_members)):
        # Run for at most 18 seconds per step to stay within timeout
        if time.time() - step_start > 18:
            print(f"Checkpointing at {idx}/{len(wav_members)} ({len(specs)} done)...")
            torch.save({"specs": specs, "labels": labels}, checkpoint_file)
            break

        m = wav_members[idx]
        label_str = m.name.strip("./").split("/")[0]
        label_idx = label_map[label_str]
        f = tar.extractfile(m)
        waveform, sr = torchaudio.load(io.BytesIO(f.read()))
        spec = extract_logmel_spectrogram(waveform, sample_rate=sr)
        specs.append(spec)
        labels.append(label_idx)
    else:
        # Loop completed naturally without break
        print("All 4890 audio samples precomputed!")
        all_specs_tensor = torch.stack(specs)
        all_labels_tensor = torch.tensor(labels, dtype=torch.long)
        torch.save({"specs": all_specs_tensor, "labels": all_labels_tensor}, cache_file)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print(f"Saved final cache to {cache_file}!")