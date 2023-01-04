import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image


st.write('Hello')


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print(input_details)
output_details = interpreter.get_output_details()


all_class = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']



def get_class(img_path):
    image_path = img_path
    img = Image.open(img_path).convert('RGB')
    open_cv_image = np.array(img)
 
    img = open_cv_image[:,:,::-1].copy()
    img = cv2.resize(img,(224,224))

    #img = Image.open(img_path)
    #img = img.resize((224,224))


    input_shape = input_details[0]['shape']
    input_data = np.array(np.expand_dims(img,0),dtype = np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    i = np.argmax(output_data)
    return all_class[i]

image = st.file_uploader('Enter Image')

if image != None:
    #print(image)
    display = Image.open(image)
    st.image(display, caption = get_class(image))
    #st.write(get_class(image))










