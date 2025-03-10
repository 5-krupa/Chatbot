import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load the intents file
file_path = r"C:\Users\Krupa.Gandhi\Downloads\chatbot\intents.json"  # Use raw string (r"")
with open(file_path, "r") as file:
    intents = json.load(file)
    print("✅ File loaded successfully!")


# Initialize empty lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each pattern in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern.lower())  # Convert to lowercase for consistency
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Remove duplicates
classes = sorted(set(classes))

print(f"Vocabulary size: {len(words)} words")
print(f"Number of classes: {len(classes)}")
print(f"Number of documents: {len(documents)}")

# Save preprocessed data
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words for each document
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create the bag of words vector
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create output row with 1 at index of current tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    # Add bag of words + output to training data
    training.append(bag + output_row)

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Split into X and Y values
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

print(f"Training data shape: X={train_x.shape}, Y={train_y.shape}")

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Define optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
history = model.fit(train_x, train_y, epochs=200, batch_size=16, verbose=1)

# Save model
model.save('chatbot_model.h5')
print("Model trained and saved successfully!")