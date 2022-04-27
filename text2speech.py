import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
# from datasets import load_dataset
import librosa

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")


# ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
audio_file = librosa.load('Planning 10.wav', sr=16000)

inputs = processor(audio_file[0], sampling_rate=audio_file[1], return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids)
print(transcription)