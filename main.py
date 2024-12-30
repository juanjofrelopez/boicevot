import speech_recognition as sr
from faster_whisper import WhisperModel 
from src.db import PgVectorConnection
from src.embeddings import TextEmbedder
from src.llm import LLMEngine
from src.utils import adapt_audio_for_transcription
import pyttsx3
import time

embedder = TextEmbedder()
db = PgVectorConnection()
llm = LLMEngine()

def process_query():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
    adapted_audio = adapt_audio_for_transcription(raw_data, sample_rate=16000, sample_width=2)

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8",cpu_threads=8)

    segments, info = model.transcribe(adapted_audio, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    query = []

    for segment in segments:
        query.append(segment.text)
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    k=60
    query = " ".join(query)
    emb = embedder.generate_embeddings_from_string(query)
    results = db.find_similar(query, emb, k)
    context = ";".join([row[2] for row in results if row[2].strip()])
    response_stream = llm.ask(query, context)

    response = []
    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            response.append(chunk.choices[0].delta.content)

    response = "".join(response)
    print(response)
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
    engine.stop()
    time.sleep(1)


def main():
    print("\n🚗 Welcome to Skibidi toilet Assistant!")
    print("------------------------------------------------")

    while True:
        try:
            print("\n❓ speak to me: ")
            process_query()
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()


