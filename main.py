import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
from spellchecker import SpellChecker
import spacy
import string
import contractions
import re  # Regex for cleaning
from googletrans import Translator

# Cache the model for faster loading
@st.cache_resource
def load_grammar_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Load the Hindi-English translation model
@st.cache_resource
def load_hindi_translator():
    model_path = "./hindi_english_translator_final"
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    return tokenizer, model

# Load the English-Hindi translation model
@st.cache_resource
def load_english_to_hindi_translator():
    model_path = "./english_to_hindi_translator_final"
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model

# Initialize Google translator (as backup)
@st.cache_resource
def load_google_translator():
    return Translator()

# Translate text from English to Hindi using our trained model
def translate_to_hindi(text):
    try:
        # Expand contractions first
        text = expand_contractions(text)
        
        # Load our trained model
        tokenizer, model = load_english_to_hindi_translator()
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        # Decode the translation
        hindi_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        # Clean up the translation
        hindi_text = hindi_text.strip()
        if hindi_text.startswith(","):
            hindi_text = hindi_text[1:].strip()
            
        return hindi_text
    except Exception as e:
        # Fallback to Google Translate if our model fails
        st.warning(f"Using backup translator due to error: {str(e)}")
        translator = load_google_translator()
        translation = translator.translate(text, dest='hi')
        return translation.text

# Grammar correction using T5 model
def correct_grammar(text, tokenizer, model):
    input_text = "fix: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected_text = final_cleanup(corrected_text)
    
    return corrected_text

# Spelling correction using pyspellchecker
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    corrections = []
    
    for word in words:
        correction = spell.correction(word)
        if correction and correction != word:
            corrections.append(f"Spelling corrected: {word} -> {correction}")
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words), corrections

# Final cleanup to remove unwanted repetitions
def final_cleanup(text):
    text = ' '.join(text.split())  # Remove extra spaces
    text = text.replace(" .", ".")  # Fix misplaced periods
    text = re.sub(r"(fix:\s*)+", "", text).strip()  # Remove repeated "fix:"
    
    return text

# Expand contractions (e.g., "I'm" -> "I am")
def expand_contractions(text):
    return contractions.fix(text)

# Extract parts of speech using spaCy
def extract_parts_of_speech(sentence):
    nlp = spacy.load("en_core_web_sm")
    sentence = expand_contractions(sentence)
    doc = nlp(sentence)
    
    pos_tags = {
        "verbs": [],
        "auxiliary_verbs": [],
        "nouns": [],
        "pronouns": [],
        "adjectives": [],
        "adverbs": []
    }
    
    for token in doc:
        if token.pos_ == 'VERB':
            pos_tags["verbs"].append(token.text)
        elif token.pos_ == 'AUX':
            pos_tags["auxiliary_verbs"].append(token.text)
        elif token.pos_ == 'NOUN':
            pos_tags["nouns"].append(token.text)
        elif token.pos_ == 'PRON':
            pos_tags["pronouns"].append(token.text)
        elif token.pos_ == 'ADJ':
            pos_tags["adjectives"].append(token.text)
        elif token.pos_ == 'ADV':
            pos_tags["adverbs"].append(token.text)
    
    return pos_tags

# Correct text logic
def correct_text(text, tokenizer, model):
    spell_corrected, spell_explanations = correct_spelling(text)
    spell_corrected = final_cleanup(spell_corrected)
    
    if len(spell_corrected.split()) == 1:
        grammar_corrected = spell_corrected  # Skip grammar correction for single words
    else:
        grammar_corrected = correct_grammar(spell_corrected, tokenizer, model)
    
    grammar_corrected = final_cleanup(grammar_corrected)
    
    explanations = spell_explanations.copy()
    if grammar_corrected != spell_corrected:
        explanations.append("Grammar corrected: Sentence structure altered.")
    
    pos = extract_parts_of_speech(grammar_corrected)
    
    return grammar_corrected, explanations, pos
