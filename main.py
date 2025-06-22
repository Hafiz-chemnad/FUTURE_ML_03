from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import os

# --- NLTK Data Download (ensure these are available on the server too) ---
# This block attempts to download NLTK data if not found or corrupted.
# It's good to have in main.py for deployment environments.
print("Checking NLTK data...")
try:
    # Explicitly download punkt and wordnet, forcing re-download if corrupted
    # Use the correct exception class for NLTK download errors
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK data (punkt, wordnet, averaged_perceptron_tagger) ensured to be downloaded.")
except Exception as e: # Catching general Exception for robustness during download
    print(f"Warning: Could not automatically download NLTK data. Please ensure you have internet access and try again. Error: {e}")
    print("If you continue to face issues, manually run: `python -m nltk.downloader punkt wordnet averaged_perceptron_tagger` in your terminal.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Load saved components ---
MODEL_PATH = "chatbot_model.h5"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"
INTENTS_PATH = "intents.json"

try:
    # Ensure files exist before attempting to load
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
    if not os.path.exists(WORDS_PATH): raise FileNotFoundError(f"Words file '{WORDS_PATH}' not found.")
    if not os.path.exists(CLASSES_PATH): raise FileNotFoundError(f"Classes file '{CLASSES_PATH}' not found.")
    if not os.path.exists(INTENTS_PATH): raise FileNotFoundError(f"Intents file '{INTENTS_PATH}' not found.")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(WORDS_PATH, 'rb') as handle:
        words = pickle.load(handle)
    with open(CLASSES_PATH, 'rb') as handle:
        classes = pickle.load(handle)
    with open(INTENTS_PATH, 'r') as f:
        intents = json.load(f)
    
    print("Chatbot components loaded successfully!")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Missing required chatbot file. {e}")
    print("Please ensure all files (chatbot_model.h5, words.pkl, classes.pkl, intents.json) are in the same directory as main.py.")
    exit() # Exit the application if critical files are missing
except Exception as e:
    print(f"An unexpected error occurred while loading chatbot components: {e}")
    exit()

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to clean up and lemmatize sentences for prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Helper function to create bag of words for a new sentence
def bag_of_words(sentence, words_vocab):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_vocab)
    for s in sentence_words:
        for i, w in enumerate(words_vocab):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Prediction function
def predict_intent(sentence):
    p = bag_of_words(sentence, words)
    # Reshape p to be a 2D array: (1, num_features)
    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD = 0.70 # Confidence threshold for an intent to be considered a match. Adjust as needed.
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(round(r[1], 4))})
    
    if not return_list:
        # If no intent is above the threshold, return the top predicted intent
        # only if its probability is reasonably low, otherwise provide a "no_match"
        top_intent_idx = np.argmax(res)
        top_intent_prob = res[top_intent_idx]
        print(f"Debug: No intent above threshold. Top intent: {classes[top_intent_idx]} with prob {top_intent_prob:.4f}")
        
        # If even the top intent is very low confidence, consider it a "no_match"
        if top_intent_prob < 0.2: # You can adjust this threshold for "unknown" queries
             return [{"intent": "no_match", "probability": "1.0"}]
        else:
             # Otherwise, return the top intent even if it's below the primary threshold
             return [{"intent": classes[top_intent_idx], "probability": str(round(top_intent_prob, 4))}]

    return return_list

# Get a random response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    
    for intent_data in intents_json['intents']:
        if intent_data['tag'] == tag:
            return random.choice(intent_data['responses'])
    
    # Fallback for "no_match" or if a tag isn't found (shouldn't happen with error handling)
    if tag == "no_match":
        return "I'm sorry, I don't quite understand that. Could you please rephrase or provide more details?"
    return "It seems I'm having trouble processing that request. Please try again or contact support."


# --- FastAPI Endpoints ---

# Root endpoint to serve the static HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found in static/ directory.</h1>", status_code=404)


# Chat endpoint to handle user messages and return bot responses
@app.post("/chat/")
async def chat(user_message: str = Form(...)):
    print(f"Received message: {user_message}")
    intents_list = predict_intent(user_message)
    print(f"Predicted intents: {intents_list}")

    response_message = get_response(intents_list, intents)
    
    return {"response": response_message, "predicted_intent": intents_list[0]['intent']}

# Optional: A simple GET endpoint for quick API testing (e.g., in browser directly)
@app.get("/test_chat")
async def test_chat_get(query: str):
    intents_list = predict_intent(query)
    response_message = get_response(intents_list, intents)
    return {"query": query, "predicted_intent": intents_list[0]['intent'], "response": response_message}