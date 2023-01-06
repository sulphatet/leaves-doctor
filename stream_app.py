import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image


st.title('The Plant Doctor: Diagnose your plants')
st.subheader('Take a photograph with your camera...')
st.subheader('...Or send an image fron your gallery to our model')


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model(1).tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print(input_details)
output_details = interpreter.get_output_details()


all_class = ['Bacterial Spots', 
                 'Black Rot', 
                 'Early Blight', 
                 'Esca (Black_Measels)', 
                 'Gray Leaf spot', 
                 'Haunglongbing', 
                 'Healthy', 
                 'Late Blight', 
                 'Leaf Mold', 
                 'Leaf Scorch', 
                 'Late Blight', 
                 'Mosaic Virus', 
                 'Northern Leaf Blight', 
                 'Powdery Mildew', 
                 'Rust', 
                 'Scab', 
                 'Septoria leaf spot', 
                 'Spider Mite', 
                 'Target Spot', 
                 'Yellow Leaf Curl Virus']


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

with st.sidebar:
    medium = st.radio(
                "Select Your image",
                ("Using Camera","Using Gallery")
            )

if medium == "Using Gallery":
    image = st.file_uploader('Enter Image')
    if image != None:
        display = Image.open(image)
        st.image(display)

if medium == "Using Camera":
    image = st.camera_input('Capture image')

if image != None:
    #print(image)
    display = Image.open(image)
    #st.image(display)
    image_class = get_class(image)
    disease = image_class
    #desease = disease.replace('_',' ')
    if disease == 'Healthy':
        statement = f'Great news! Your plant is healthy'
    else:
        statement = f'Uh oh....Your plant appears to have {disease}'
    st.subheader(statement)
    #st.write(get_class(image))

