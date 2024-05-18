import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO

# Load the pre-trained model
model = load_model(r'C:\ImageClassification\Image_classify.keras')

# List of categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
            'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
            'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 
            'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 
            'tomato', 'turnip', 'watermelon']

# Image dimensions
img_width, img_height = 180, 180

# Directory containing the example images for each category
example_images_dir = r'C:\ImageClassification\ExampleImages'  # Adjust the path to where your example images are stored

# Streamlit app
st.title('Image Classification App')

# About section
st.sidebar.title("About")
st.sidebar.info("This app classifies images of vegetables and fruits. You can upload an image, provide a URL, or enter the path to a local file to get the classification result along with the top predictions.")

# User input for image path or URL
image_source = st.sidebar.radio("Select Image Source", ("Upload", "URL", "Local Path"))

if image_source == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_load = Image.open(uploaded_file)
        image_path = uploaded_file.name
elif image_source == "URL":
    image_url = st.text_input('Enter the image URL')
    if image_url:
        response = requests.get(image_url)
        image_load = Image.open(BytesIO(response.content))
        image_path = image_url
else:
    image_path = st.text_input('Enter the path to the image', r'C:\ImageClassification\Apple.jpg')
    if image_path:
        image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))

if 'image_load' in locals():
    # Resize image if necessary
    image_load = image_load.resize((img_width, img_height))
    
    # Convert image to array
    img_arr = tf.keras.preprocessing.image.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    # Predict the category of the image
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Get the predicted category and confidence score
    predicted_index = np.argmax(score)
    predicted_category = data_cat[predicted_index]

    # Display the input image and the prediction result
    st.image(image_load, caption="Input Image", width=200)
    st.write(f'Veg/Fruit in image is {predicted_category}')
    
    # Load and display the correctly predicted image
    predicted_image_path = os.path.join(example_images_dir, f'{predicted_category}.jpg')
    
