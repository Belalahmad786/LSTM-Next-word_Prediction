import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Load trained model
model = load_model("nextword_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Reverse word index
reversed_index = {idx: word for word, idx in tokenizer.word_index.items()}

# Max sequence length
max_len = 44

# Text generation function
def generate_text(seed_text, num_words=10):

    text = seed_text

    for _ in range(num_words):

        seq = tokenizer.texts_to_sequences([text])[0]

        padded = pad_sequences([seq], maxlen=max_len - 1, padding='pre')

        preds = model.predict(padded, verbose=0)

        predicted_index = np.argmax(preds)

        next_word = reversed_index.get(predicted_index, "")

        text += " " + next_word

    return text


# Streamlit UI
st.title("Next Word Prediction with Deep Learning")

seed = st.text_input("Enter a starting text:", "Hello")

num_words = st.slider("Number of words to generate", 1, 20, 10)

if st.button("Generate"):

    result = generate_text(seed, num_words)

    st.success(result)