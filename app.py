import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal

# Load the model
custom_objects = {'Orthogonal': Orthogonal}
model = load_model('toxicity.h5', custom_objects=custom_objects)

# Load the dataset to access column names
df = pd.read_csv('train.csv')

# Tokenize comments
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(df['comment_text'])
MAX_LENGTH = 1800  # Same as before

def score_comment(comment):
    # Tokenize the comment
    sequences = tokenizer.texts_to_sequences([comment])
    # Pad sequences to fixed length
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH)
    # Predict using the model
    results = model.predict(padded_sequences)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text

# Create Streamlit interface
st.title('Toxic Comment Classifier')
comment = st.text_area('Enter your comment here:', height=200)
if st.button('Classify'):
    prediction = score_comment(comment)
    st.text(prediction)
