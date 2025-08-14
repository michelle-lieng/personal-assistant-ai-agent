import os
import sounddevice as sd
import pyttsx3
import google.generativeai as genai
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=(
                "You are a silly goofy assistant who speaks succintly. "
                "Answer in 1–2 sentences."
            ),
            generation_config={
                            "temperature": 0.7,         # creativity (0.0 = deterministic, 1.0 = more random)
                            "max_output_tokens": 70,    # ~rough max words/tokens, 1 token ≈ 4 chars in English => 200 characters
                            "top_p": 0.9,               # nucleus sampling
                            "top_k": 40                 # limits sampling pool
                            })

def select_input_device():
    """
    Scans to find a microphone device.
    """
    devices = sd.query_devices()
    input_ids = [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    if not input_ids:
        raise RuntimeError("No input (microphone) devices found. Check Windows mic permissions.")
    # Prefer something that looks like a microphone if possible
    for i in input_ids:
        name = devices[i]["name"]
        if "mic" in name.lower() or "microphone" in name.lower():
            sd.default.device = (i, None)  # (input_id, output_id)
            print(f"Using input device: [{i}] {name}")
            return
    # Fallback: first input device
    i = input_ids[0]
    sd.default.device = (i, None)
    print(f"Using default input device: [{i}] {devices[i]['name']}")

SR = 16000
def record(seconds=5):
    # turns “record 5 seconds” into an exact sample count (e.g., 5 × 16000 = 80000 samples).
    frames = int(seconds * SR)
    print("Speak now...")
    # records exactly frames amount of time-steps of audio from your mic and puts them into a NumPy array. 
    data = sd.rec(frames, samplerate=SR, channels=1, dtype="float32")
    # Then you call sd.wait() to pause until the recording is finished.
    sd.wait()  
    print("Recorded!!!!!!!")
    return data[:, 0]  # float32 in [-1, 1]

def transcribe(audio_f32):
    """
    This function turns audio into text.

    The whisper transcribe model keeps several candidate transcriptions and picks the one that 
    the model thinks is most probable (i.e., “makes the most sense”) given the audio 
    and language patterns it learned.
    """
    # configure the model
    whisper = WhisperModel("small", compute_type="int8")  # small = faster for testing

    segments, info = whisper.transcribe(
        audio_f32, # this is your microphones recording
        language="en", # this says to treat audio like english
        beam_size=5, # means it keeps 5 parallel guesses at every step and chooses the overall most probable one at the end 
        vad_filter=False # don't auto-trim silence - transcripts the whole clip
    )
    #print(f"Segments: {[seg for seg in segments]}")
    text = "".join(seg.text for seg in segments).strip()
    print(f"Your transcribed input: {text}")
    return text

def speak(text):
    """
    This function turns text into speech.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main(seconds=5):
    # STEP 1: Select input device
    select_input_device()
    
    # STEP 2: Record what you say for 5 seconds
    audio = record(seconds)

    # STEP 3: Get's what you said and transcribes it to text
    user_text = transcribe(audio)
    if not user_text:
        print("No speech detected.")
        return
    
    # STEP 4: Uses gemini to generate a reply
    reply = model.generate_content(user_text).text
    print(f"Gemini: {reply}")

    # STEP 5: Turns text into speech and replies to you
    speak(reply)

if __name__ == "__main__":
    main(5)
