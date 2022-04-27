from huggingsound import SpeechRecognitionModel
from transformers import AutoTokenizer, AutoModelWithLMHead

doctor_tokenizer = AutoTokenizer.from_pretrained("flexudy/t5-base-multi-sentence-doctor")
doctor_model = AutoModelWithLMHead.from_pretrained("flexudy/t5-base-multi-sentence-doctor")

model = SpeechRecognitionModel("facebook/wav2vec2-large-960h-lv60-self")
audio_paths = ["eng_m4.wav", "Planning 10.wav"]

transcriptions = model.transcribe(audio_paths)

for t in transcriptions:
    transcription = t['transcription']
    print()
    print("-----------------")
    print(transcription)
    print()
    
    # Attempt to repair the sentence
    input_text = "repair_sentence: " + transcription + " context: " + transcription
    
    input_ids = doctor_tokenizer.encode(input_text, return_tensors="pt")

    outputs = doctor_model.generate(input_ids, max_length=32, num_beams=1)

    sentence = doctor_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("Repaired sentence: \n" + sentence)