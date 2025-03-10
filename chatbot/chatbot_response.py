import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents, words, classes, and model
file_path = r"C:\Users\Krupa.Gandhi\Downloads\chatbot\intents.json"  # Use raw string (r"")
with open(file_path, "r") as file:
    intents = json.load(file)
    print("✅ File loaded successfully!")


try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
    print("✅ Model loaded successfully!")
    print(f"Vocabulary size: {len(words)} words")
    print(f"Number of classes: {len(classes)}")
except Exception as e:
    print(f"Error loading model or data: {e}")
    exit(1)

def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence"""
    sentence_words = nltk.word_tokenize(sentence.lower())  # Convert to lowercase
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Convert a sentence to a bag of words"""
    sentence_words = clean_up_sentence(sentence)
    print(f"Processed Words: {sentence_words}")
    
    # Create bag of words array
    bag = [0] * len(words)
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if word == s_word:
                bag[i] = 1
    
    print(f"Bag of Words: {np.array(bag)}")
    return np.array(bag)

def predict_class(sentence):
    """Predict the class of a sentence"""
    # Generate bag of words
    bow = bag_of_words(sentence)
    
    # Check if bag is all zeros
    if np.sum(bow) == 0:
        print("Warning: No words matched vocabulary!")
    
    # Predict using model
    res = model.predict(np.array([bow]))[0]
    
    # Filter out predictions below threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # If no prediction above threshold
    if not results:
        return [{"intent": "unknown", "probability": "0.0"}]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return list of intents and probabilities
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    """Get a response based on the predicted intent"""
    if intents_list[0]['intent'] == "unknown":
        return "I'm not sure what you mean. Could you rephrase that?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    else:  # No matching intent found
        result = "I'm not sure how to respond to that."
    
    return result

# Run the chatbot
print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    
    ints = predict_class(message)
    print(f"Predicted Intent: {ints}")
    resp = get_response(ints, intents)
    print(f"Bot: {resp}")