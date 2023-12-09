import streamlit as st
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from PIL import Image

# Load the model
loaded_model = tf.keras.models.load_model(r'C:\Users\hp\Pictures\Malaria_Cells.h5', compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels (replace this with your actual class labels)
classes = ['Parasitized', 'Uninfected']  # Replace with actual class labels

# Function to make predictions
def predict_malaria(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = classes[tf.argmax(score)]
    return predicted_class, score

# Streamlit app
st.title('Malaria Cell Image Prediction')
st.write('Upload an image for malaria prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Predict'):
        predicted_class, score = predict_malaria(image)
        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {100 * tf.reduce_max(score):.2f}%')
