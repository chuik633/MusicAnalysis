import pickle
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import librosa 

mapped_instruments = {1: 'Piano', 7: 'Harpsichord', 41: 'Violin, fiddle', 43: 'Cello', 61: 'French horn', 72: 'Clarinet', 74: 'Flute'}

# Load the yamnet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the trained model, scaler, and label encoder
with open('./instrumentClassification/trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)


def classify_instrument(audio_path):

    # extract features using yamnet
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(audio_tensor)
    features = embeddings.numpy().mean(axis=0)  # Average across time

    # Predict using  trained model
    y_pred = trained_model.predict([features])[0].tolist()
    print('pyred', y_pred)
    print(y_pred.__class__)

    # Map predictions to instrument names
    instruments = [list(mapped_instruments.values())[i] for i, val in enumerate(y_pred) if val == 1]
    return instruments

