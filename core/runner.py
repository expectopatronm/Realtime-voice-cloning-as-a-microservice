from pathlib import Path

import numpy as np
import torch

from core.voice_cloner.encoder import inference as encoder
from core.voice_cloner.synthesizer.inference import Synthesizer
from core.voice_cloner.utils.default_models import ensure_default_models
from core.voice_cloner.vocoder import inference as vocoder

enc_model_fpath = Path("core/voice_cloner/saved_models/default/encoder.pt")
syn_model_fpath = Path("core/voice_cloner/saved_models/default/synthesizer.pt")
voc_model_fpath = Path("core/voice_cloner/saved_models/default/vocoder.pt")

sample_1 = "core/voice_cloner/samples/1320_00000.mp3"
in_fpath = Path(sample_1)

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
        "%.1fGb total memory.\n" %
        (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
ensure_default_models(Path("core/voice_cloner/saved_models"))
encoder.load_model(enc_model_fpath)
synthesizer = Synthesizer(syn_model_fpath)
vocoder.load_model(voc_model_fpath)


def synthesize_speech(audio_file):

    text = "Have you been a naughty boy?"

    ## Computing the embedding
    preprocessed_wav = encoder.preprocess_wav(audio_file)
    print("Loaded file succesfully")

    # Then we derive the embedding.
    embed = encoder.embed_utterance(preprocessed_wav)
    print(type(embed))
    print("Created the embedding")

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]

    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")

    ## Generating the waveform
    print("Synthesizing the waveform:")

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)

    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # # Save it on the disk
    # filename = "demo_output.wav"
    # print(generated_wav.dtype)
    # sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    # print("\nSaved output as %s\n\n" % filename)

    return generated_wav, synthesizer.sample_rate