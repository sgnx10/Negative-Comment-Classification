import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Load the model
model = tf.keras.models.load_model('toxicity.h5')

# Load the dataset to access column names
df = pd.read_csv('train.csv')

# Initialize TextVectorization
MAX_FEATURES = 200000
MAX_LENGTH = 1800
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=MAX_LENGTH,
                               output_mode='int')
vectorizer.adapt(df['comment_text'].values)


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

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
