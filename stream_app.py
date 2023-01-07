import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image


#st.title('The Plant Doctor: Diagnose your plants')
#st.subheader('Take a photograph with your camera...')
#st.subheader('...Or send an image fron your gallery to our model')


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model(2).tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print(input_details)
output_details = interpreter.get_output_details()


all_class =  ['Bacterial Spots',
              'Black Rot',
              'Esca (Black_Measels)',
              'Gray Leaf Spots',
              'Haunglongbing',
              'Healthy',
              'Leaf Blight',
              'Leaf Mold',
              'Leaf Scorch',
              'Mosaic Virus',
              'Powdery Mildew',
              'Rust',
              'Scab',
              'Septoria Leaf Spots',
              'Spider Mite',
              'Target Spots',
              'Yellow Leaf Curl Virus']

all_links =  {'Bacterial Spots' : 'https://www.gardeningknowhow.com/plant-problems/disease/bacterial-leaf-spot.htm',
              'Black Rot' : 'https://www.gardeningknowhow.com/plant-problems/disease/black-rot-of-cole-crops.htm',
              'Esca (Black_Measels)' : 'https://grapes.extension.org/grapevine-measles/',
              'Gray Leaf Spots' : 'https://www.gardeningknowhow.com/plant-problems/disease/plant-leaf-spots.htm',
              'Haunglongbing' : 'https://ucanr.edu/sites/Citrus@UCR/Huanglongbing/',
              'Healthy' : None,
              'Leaf Blight' : 'https://www.homequestionsanswered.com/what-is-leaf-blight.htm',
              'Leaf Mold' : 'https://u.osu.edu/vegetablediseasefacts/tomato-diseases/high-tunnel-diseases/leaf-mold/',
              'Leaf Scorch' : 'https://www.gardeningknowhow.com/ornamental/trees/tgen/bacterial-leaf-scorch-disease.htm',
              'Mosaic Virus' : 'https://www.almanac.com/pest/mosaic-viruses',
              'Powdery Mildew' : 'https://www.almanac.com/pest/powdery-mildew',
              'Rust' : 'https://smartgardenguide.com/rust-spots-on-leaves/',
              'Scab' : 'https://www.gardeningknowhow.com/edible/vegetables/vgen/scab-on-vegetables.htm',
              'Septoria Leaf Spots' : 'https://www.gardeningknowhow.com/edible/fruits/berries/septoria-cane-leaf-spot-disease.htm',
              'Spider Mite' : 'https://www.cannagardening.com/spider-mite-pests-diseases',
              'Target Spots' : 'https://guide.utcrops.com/cotton/cotton-foliar-diseases/target-spot/',
              'Yellow Leaf Curl Virus' : 'https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/tomato-yellow-leaf-curl-virus'
             }

def get_class(img_path):
    image_path = img_path
    img = Image.open(img_path).convert('RGB')
    open_cv_image = np.array(img)
 
    img = open_cv_image[:,:,::-1].copy()
    img = cv2.resize(img,(224,224))

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
    cure = all_links[disease]
    if cure is not None:
        cure = f'To know more about {disease} and how to treat it, you can visit: {cure}'
        st.markdown(cure)
    #st.write(get_class(image))
