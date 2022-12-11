import numpy as np

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder

SAMPLE_RATE = 22050
record_seconds = 10
embedding = None

text = "Are you being a naughty boy?" 

def _compute_embedding(audio):

  global embedding

  embedding = None
  embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, SAMPLE_RATE))

def _retrain_on_audio(b):

  audio = upload_audio(sample_rate=SAMPLE_RATE)
  _compute_embedding(audio)

def synthesize(embed, text):

  print("Synthesizing new audio...")
  specs = synthesizer.synthesize_spectrograms([text], [embed])
  generated_wav = vocoder.infer_waveform(specs[0])
  generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

if embedding is None:
  print("First retain on custom audio.")
else:
  synthesize(embedding, text)