import streamlit as st
from main import load_grammar_model, correct_text, extract_parts_of_speech
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch
from transformers import pipeline
from spellchecker import SpellChecker
import spacy
import contractions
import numpy as np
import sounddevice as sd
import wave
from faster_whisper import WhisperModel
import os

# Initialize session state variables
if 'original_text' not in st.session_state:
    st.session_state['original_text'] = ""
if 'corrected_text' not in st.session_state:
    st.session_state['corrected_text'] = ""
if 'show_correction' not in st.session_state:
    st.session_state['show_correction'] = False
if 'explanations' not in st.session_state:
    st.session_state['explanations'] = []

# Load all models at startup
@st.cache_resource
def load_all_models():
    # Load grammar model
    tokenizer, model = load_grammar_model()
    
    # Load Hindi translation model
    hindi_model_path = "./english_to_hindi_translator_final"
    hindi_tokenizer = MarianTokenizer.from_pretrained(hindi_model_path)
    hindi_model = MarianMTModel.from_pretrained(hindi_model_path)
    hindi_model.eval()
    
    # Load Japanese translation model
    japanese_model_path = "./english_to_japanese_translator_final"
    japanese_tokenizer = AutoTokenizer.from_pretrained(japanese_model_path)
    japanese_model = AutoModelForSeq2SeqLM.from_pretrained(japanese_model_path)
    japanese_model.eval()
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize spell checker
    spell = SpellChecker()
    
    # Initialize Whisper model for transcription
    whisper_model = WhisperModel("large-v1", device="cpu", compute_type="float32")
    
    return {
        'grammar_tokenizer': tokenizer,
        'grammar_model': model,
        'hindi_tokenizer': hindi_tokenizer,
        'hindi_model': hindi_model,
        'japanese_tokenizer': japanese_tokenizer,
        'japanese_model': japanese_model,
        'nlp': nlp,
        'spell': spell,
        'whisper_model': whisper_model
    }

# Load all models
models = load_all_models()

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'

def record_audio(duration):
    """Record audio for specified duration"""
    try:
        # Check if default input device is available
        default_device = sd.default.device[0]
        if default_device is None:
            raise Exception("No default microphone found")
            
        frames = sd.rec(int(SAMPLE_RATE * duration), 
                       samplerate=SAMPLE_RATE, 
                       channels=CHANNELS, 
                       dtype=DTYPE)
        sd.wait()
        return frames
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        st.error("Please check if your microphone is connected and enabled.")
        return None

def save_audio(frames, filename):
    """Save recorded audio to WAV file"""
    if frames is None:
        st.error("No audio data to save!")
        return False
        
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(frames.tobytes())
        return True
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return False

def transcribe_audio(model, file_path):
    """Transcribe audio file using Whisper model"""
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

def translate_to_hindi(text):
    """Translate text to Hindi using our trained model"""
    try:
        inputs = models['hindi_tokenizer'](text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            translated = models['hindi_model'].generate(**inputs)
        hindi_text = models['hindi_tokenizer'].batch_decode(translated, skip_special_tokens=True)[0]
        
        # Clean up the translation
        hindi_text = hindi_text.strip()
        if hindi_text.startswith(","):
            hindi_text = hindi_text[1:].strip()
            
        return hindi_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return "Translation failed. Please try again."

def translate_to_japanese(text):
    """Translate text to Japanese using our trained model"""
    try:
        inputs = models['japanese_tokenizer'](text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        forced_bos_token_id = models['japanese_tokenizer'].lang_code_to_id["jpn_Jpan"]
        with torch.no_grad():
            translated = models['japanese_model'].generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
        japanese_text = models['japanese_tokenizer'].batch_decode(translated, skip_special_tokens=True)[0]
        return japanese_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return "Translation failed. Please try again."

# Streamlit interface
st.title("📝 Grammar and Spelling Correction Tool")
st.write("Enter a sentence below to correct grammatical and spelling errors.")

# Create a container for text input and recording
input_container = st.container()

with input_container:
    # Create columns for text input and mic button
    col1, col2 = st.columns([0.9, 0.1])
    
    with col1:
        text_input = st.text_area("Enter text to correct:", value=st.session_state['original_text'], height=100)
        if text_input != st.session_state['original_text']:
            st.session_state['original_text'] = text_input
            st.session_state['show_correction'] = False
            st.rerun()
    
    with col2:
        st.write("")  # Add some space
        st.write("")  # Add some space
        # Create a placeholder for recording status
        status_placeholder = st.empty()
        if st.button("🎤", help="Click to record voice (5 seconds)"):
            try:
                # Show recording status
                with status_placeholder.container():
                    st.markdown(
                        """
                        <style>
                        .recording-dot {
                            color: #4CAF50;
                            font-size: 24px;
                            animation: blink 1s infinite;
                        }
                        @keyframes blink {
                            50% { opacity: 0; }
                        }
                        .stopped-dot {
                            color: #f44336;
                            font-size: 24px;
                        }
                        .loading-text {
                            color: #2196F3;
                            font-size: 16px;
                        }
                        </style>
                        <div style="text-align: center">
                            <span class="recording-dot">●</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    audio_frames = record_audio(5)
                    
                    if audio_frames is not None:
                        if save_audio(audio_frames, "temp_audio.wav"):
                            # Show loading text
                            st.markdown(
                                """
                                <div style="text-align: center">
                                    <span class="loading-text">Loading...</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            transcribed_text = transcribe_audio(models['whisper_model'], "temp_audio.wav")
                            st.session_state['original_text'] = transcribed_text
                            st.session_state['show_correction'] = False
                            
                            # Show stopped indicator after loading
                            st.markdown(
                                """
                                <div style="text-align: center">
                                    <span class="stopped-dot">⬤</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            os.remove("temp_audio.wav")
                            st.rerun()
                        else:
                            st.error("❌")
            except Exception as e:
                status_placeholder.error("❌")

# Show original/transcribed text
if st.session_state['original_text']:
    st.subheader("Original/Transcribed Text:")
    st.write(st.session_state['original_text'])

# Process the text
if st.session_state['original_text']:
    # Correct text
    if st.button("Correct Text"):
        corrected_text, explanations, pos = correct_text(st.session_state['original_text'], models['grammar_tokenizer'], models['grammar_model'])
        st.session_state['corrected_text'] = corrected_text
        st.session_state['explanations'] = explanations
        st.session_state['show_correction'] = True
        st.rerun()
    
    # Show correction if available
    if st.session_state['show_correction']:
        st.subheader("Corrected Text:")
        st.write(st.session_state['corrected_text'])
        
        # Show parts of speech analysis
        st.subheader("Parts of Speech Analysis:")
        pos = extract_parts_of_speech(st.session_state['corrected_text'])
        for category, words in pos.items():
            if words:  # Only show categories that have words
                st.write(f"{category.replace('_', ' ').title()}:")
                for word in words:
                    st.write(f"  - {word}")
        
        # Show explanations for corrections
        if st.session_state['explanations']:
            st.subheader("Corrections Made:")
            for explanation in st.session_state['explanations']:
                st.write(f"• {explanation}")
        
        # Translation section
        st.subheader("Translation")
        translation_language = st.selectbox(
            "Select translation language:",
            ["Hindi", "Japanese"]
        )
        
        if st.button(f"Translate to {translation_language}"):
            if translation_language == "Hindi":
                translated_text = translate_to_hindi(st.session_state['corrected_text'])
                st.write("Hindi translation (using trained model):")
            else:  # Japanese
                translated_text = translate_to_japanese(st.session_state['corrected_text'])
                st.write("Japanese translation (using trained model):")
            st.write(translated_text)

st.sidebar.title("ℹ️ How to Use This App")
st.sidebar.write("""
### 🎤 Voice Recording
1. Click the microphone icon (🎤) next to the text box
2. Speak clearly into your microphone
3. Wait for the transcription to appear

### ✍️ Text Input
- Type directly in the text box
- OR use the transcribed text from voice recording

### ✅ Text Correction
1. Click "Correct Text" to:
   - Fix grammar and spelling
   - See explanations for corrections
   - View parts of speech analysis

### 🌐 Translation
1. First correct your text
2. Select your desired translation language (Hindi or Japanese)
3. Click "Translate" to see the translation
4. View the translation using our trained model

### 💡 Tips
- Speak clearly when recording
- Wait for each step to complete
- Check the corrections before translating
""")
