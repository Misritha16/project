import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!','.']

# Loop through each intent
for intent in intents['intents']:
    # Loop through each pattern in the intent
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))
        # Add intent tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(words,classes)
# Lemmatize and lowercase each word, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training set, bag of words for each sentence
for doc in documents:
    # Initialize bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word and create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output is '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle our features
random.shuffle(training)

# Split training set into X and Y
X = np.array([i[0] for i in training])
Y = np.array([i[1] for i in training])

# Pad sequences to ensure uniform length
max_seq_length = max(len(x) for x in X)
X = pad_sequences(X, maxlen=max_seq_length, padding='post')

print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(max_seq_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))  # Corrected output layer

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
hist = model.fit(X, Y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.h5', hist)
print("Model created and saved")
