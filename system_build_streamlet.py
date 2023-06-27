# Modules
import pyrebase
import streamlit as st
from datetime import datetime
####################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
import plotly.graph_objects as go
import librosa
import wave
import soundfile as sf
from tensorflow.keras.preprocessing import image
import socket
from PIL import Image
####################################



firebaseConfig = {
  'apiKey': "AIzaSyAIm1x1qL5p6SfVEa3iJtsf11Cv2SWaXEk",
  'authDomain': "pests-framework.firebaseapp.com",
  'projectId': "pests-framework",
  'databaseURL': "https://pests-framework-default-rtdb.europe-west1.firebasedatabase.app/",
  'storageBucket': "pests-framework.appspot.com",
  'messagingSenderId': "120939001207",
  'appId': "1:120939001207:web:eae4c48e955f275b7d1d72",
  'measurementId': "G-L74NJPS323"
};

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

# Database

title_container = st.container()
col1, col2 = st.columns([1, 50])


image = storage.child("gradient_pests_logo_F.png").get_url("gradient_pests_logo_F.png")

with title_container:
    with col1:
        st.image(image, width=700)



 

def main():

   #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
  
   # 2. horizontal menu
   bio = option_menu(
        None,
        ['Home', 'Classify', 'Upload', 'Users', 'Report'],
        icons=['house', 'person-circle', 'gear', 'cloud', 'cloud-upload'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        #bg_color="blue",
        #fg_color="black",
        
        #font=("Arial", 12, "bold"),
        
        #highlight_color="blue",
        #highlight_thickness=2,
        #width=200,
        #height=40,
        #border_width=1,
        #border_color="black",
        #padding=5,
    )

   if st.button("Logout"):
     # Clear session state and redirect to login page
         st.session_state.logged_in = False
         st.experimental_rerun()
   if bio == 'Report':
    all_users = db.get()
    res = []

    # Store all the users handle name
    for users_handle in all_users.each():
        k = users_handle.val()["Handle"]
        res.append(k)

    # Total users
    nl = len(res)
    st.write('Total users here: ' + str(nl))

    # Retrieve all classifications from all users
    all_classifications = []
    for users_handle in all_users.each():
        lid = users_handle.val()["ID"]
        user_classifications = db.child(lid).child("classification").get()
        if user_classifications.val() is not None:
            for classification in user_classifications.each():
                classification_val = classification.val()
                camera_id = classification_val.get("Camera_ID")
                farm_id = classification_val.get("Farm_ID")
                if camera_id is not None:
                    all_classifications.append((camera_id, farm_id))

    # Show the classification results
    if len(all_classifications) > 0:
        # Count the occurrences of each camera ID and farm ID
        camera_counts = {}
        farm_counts = {}
        for camera_id, farm_id in all_classifications:
            if camera_id in camera_counts:
                camera_counts[camera_id] += 1
            else:
                camera_counts[camera_id] = 1

            if farm_id in farm_counts:
                farm_counts[farm_id] += 1
            else:
                farm_counts[farm_id] = 1

        # Prepare data for visualization
        camera_labels = list(camera_counts.keys())
        camera_values = list(camera_counts.values())

        farm_labels = list(farm_counts.keys())
        farm_values = list(farm_counts.values())

        # Create a bar chart for camera ID
        fig_camera, ax_camera = plt.subplots()
        ax_camera.bar(camera_labels, camera_values)

        # Customize the chart for camera ID
        ax_camera.set_xlabel('Camera ID')
        ax_camera.set_ylabel('Number of Classifications')
        ax_camera.set_title('Classification Count by Camera ID')

        # Rotate x-axis labels if needed for camera ID
        plt.xticks(rotation=45)

        # Display the chart for camera ID
        st.pyplot(fig_camera)

        # Create a pie chart for camera ID
        fig_camera_pie, ax_camera_pie = plt.subplots()
        ax_camera_pie.pie(camera_values, labels=camera_labels, autopct='%1.1f%%', startangle=90)

        # Customize the chart for camera ID
        ax_camera_pie.set_title('Classification Distribution by Camera ID')

        # Display the chart for camera ID
        st.pyplot(fig_camera_pie)

        # Create a bar chart for farm ID
        fig_farm, ax_farm = plt.subplots()
        ax_farm.bar(farm_labels, farm_values)

        # Customize the chart for farm ID
        ax_farm.set_xlabel('Farm ID')
        ax_farm.set_ylabel('Number of Classifications')
        ax_farm.set_title('Classification Count by Farm ID')

        # Rotate x-axis labels if needed for farm ID
        plt.xticks(rotation=45)

        # Display the chart for farm ID
        st.pyplot(fig_farm)

        # Create a pie chart for farm ID
        fig_farm_pie, ax_farm_pie = plt.subplots()
        ax_farm_pie.pie(farm_values, labels=farm_labels, autopct='%1.1f%%', startangle=90)

        ax_farm_pie.set_title('Classification Distribution by Farm ID')

        # Display the chart for farm ID
        st.pyplot(fig_farm_pie)
    else:
        st.error('There are no classifications yet for any user')


   if bio == 'Home':
    #st.title('𝓦𝓮𝓵𝓬𝓸𝓶𝓮 𝓽𝓸 𝓘𝓹𝓮𝓼𝓽𝓼 𝓕𝓻𝓪𝓶𝓮𝔀𝓸𝓻𝓴')
    if 'email' not in st.session_state:
        st.title('mail not here')
    user = auth.sign_in_with_email_and_password(st.session_state.email,st.session_state.password)
    handle = db.child(user['localId']).child("Handle").get().val()
    st.title('Welcome ' + handle)

  # Users PAGE
   elif bio == 'Users':
        all_users = db.get()
        res = []
        # Store all the users handle name
        for users_handle in all_users.each():
            k = users_handle.val()["Handle"]
            res.append(k)
        # Total users
        nl = len(res)
        st.write('Total users here: '+ str(nl)) 
        
        # Allow the user to choose which other user he/she wants to see 
        choice = st.selectbox('please choose the user to show profile',res)
        push = st.button('Show Classification list')
        
        # Show the choosen Profile
        if push:
            for users_handle in all_users.each():
                k = users_handle.val()["Handle"]
                # 
                if k == choice:
                    lid = users_handle.val()["ID"]
                    
                    handlename = db.child(lid).child("Handle").get().val()             
                    
                    st.markdown(handlename, unsafe_allow_html=True)
                    
    
                    # All posts
                    all_posts = db.child(lid).child("classification").get()
                    if all_posts.val() is not None:    
                        #for Posts in reversed(all_posts.each()):
                            #st.code(Posts.val(),language = '')
                            #df = pd.DataFrame(all_posts)
                            #st.table(df)
                        result=all_posts.val()   
                        #df = pd.DataFrame(result)
                        test = pd.DataFrame(result).astype(str)
                        test=pd.DataFrame.transpose(test)
                        st.table(test)
                    else:
                        st.error('There is no classification yet for this user')
   elif bio == 'Classify':
       choice = st.selectbox('login/Signup', ['birds image','birds sound', 'insects','weevils'])
       # Load the trained model
       if choice=='birds image':
           def load_model():
               
               model = tf.keras.models.load_model("C:\\Users\\Dell\\application.h5")
               return model
    
           # Function to import and predict the image
           def import_and_predict(image_data, model):
               size = (224, 224)
               image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
               image = np.asarray(image)
               img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               img_reshape = img[np.newaxis, ...]
               prediction = model.predict(img_reshape)
               return prediction
    
           # Streamlit app code
    
           st.title("Birds breeds Classification")
           
           
           
    
           farmid=st.text_input("Please input farm ID")
           cameraid=st.text_input("Please input Camera ID")
    
           file = st.file_uploader('Upload a bird', type=["jpg", "png"])

           # Checkbox to enable classification
           classify = st.button('Classify')
    
           if file is None:
               st.warning("Please upload an image")
           elif classify:
               # Load the trained model
               model = load_model()
               
               # Display the uploaded image
               st.write('This is your bird image')
               image = Image.open(file)
               st.image(image, use_column_width=True)
               
               # Perform classification on the uploaded image
               predictions = import_and_predict(image, model)
               class_names = ['ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL', 'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO','AFRICAN FIREFINCH','HOUSE SPARROW']
               score = tf.nn.softmax(predictions[0])
               predicted_class = class_names[np.argmax(score)]
               confidence = 100 * np.max(score)
               
               # Display the classification result
               #st.success("This image most likely belongs to {}".format(predicted_class, confidence))
               string =(
               "This image most likely belongs to {}."
               .format(class_names[np.argmax(score)], 100 * np.max(score))
               )
               st.success(string)
               #####################firebase upload
               user = auth.sign_in_with_email_and_password(st.session_state.email,st.session_state.password)
               uid = user['localId']
               fireb_upload = storage.child(file.name).put(file.getvalue(),user['idToken'])
               a_imgdata_url = storage.child(file.name).get_url(fireb_upload['downloadTokens']) 
                   
    
               now = datetime.now()
               dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
               post = {'Farm_ID' :farmid,
                       'Camera_ID' : cameraid,
                       'Classification' : string,
                       'Time Stamp' : dt_string,
                       'Image URL' : a_imgdata_url}                           
               results = db.child(user['localId']).child("classification").push(post)
                   
               st.success('Saved in firebase') 
    ############ birds sound
       elif choice=='birds sound':
           st.title("birds Classification by sound ")

           farmid=st.text_input("Please input farm ID")
           cameraid=st.text_input("Please input mic ID")
           file=st.text_input("Please add audio path")
           #file = st.file_uploader('Upload an audio', type=["wav"])
           classify = st.button('classify')
           if file is None:
               st.warning("Please upload an audio file")
           elif classify:
           # Checkbox to enable classification
               classify = st.button('Classify')
               client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
                # Define the server address and port
               server_address = ('localhost', 8000)
                
                # Connect to the server
               client_socket.connect(server_address)
               print('Connected to the server.')
               #message2 = file.path
               #st.success(message2)
                # Send data to the server
               message = file
               client_socket.send(message.encode())
                
                # Receive the server's response
               response = client_socket.recv(1024)
               print(f'Received response: {response.decode()}')
               st.success(response.decode())
                # Close the connection
               client_socket.close()

    ###########weeeevilssssss
    
       elif choice=='weevils':
           st.title("weevils detection by sound ")

           farmid=st.text_input("Please input farm ID")
           micid=st.text_input("Please input mic ID")
           file=st.text_input("Please add audio path")
           #file = st.file_uploader('Upload an audio', type=["wav"])
           classify = st.button('classify')
           if file is None:
               st.warning("Please upload an audio file")
           elif classify:
           # Checkbox to enable classification
               classify = st.button('Classify')
               client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
                # Define the server address and port
               server_address = ('localhost', 8000)
                
                # Connect to the server
               client_socket.connect(server_address)
               print('Connected to the server.')
               #message2 = file.path
               #st.success(message2)
                # Send data to the server
               message = file
               client_socket.send(message.encode())
                
                # Receive the server's response
               response = client_socket.recv(1024)
               print(f'Received response: {response.decode()}')
               st.success(response.decode())
                # Close the connection
               client_socket.close()

############################insects
       if choice=='insects':
           def load_model():
               
               model = tf.keras.models.load_model("C:\\Users\\Dell\\Inceptionv3_model_08-0.89.h5")
               return model
    
           # Function to import and predict the image
           def import_and_predict(image_data, model):
               size = (300, 300)
               image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
               image = np.asarray(image)
               img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
               #img_reshape = img[np.newaxis, ...]
               prediction = model.predict(img)
               return prediction
    
           # Streamlit app code
    
           st.title("insects breeds Classification")
           
           
           
    
           farmid=st.text_input("Please input farm ID")
           cameraid=st.text_input("Please input Camera ID")
    
           file = st.file_uploader('Upload a bird', type=["jpg", "png"])
    
           # Checkbox to enable classification
           classify = st.button('Classify')
    
           if file is None:
               st.warning("Please upload an image")
           elif classify:
               # Load the trained model
               model = load_model()
               
               # Display the uploaded image
               st.write('This is your bird image')
               image = Image.open(file)
               st.image(image, use_column_width=True)
               image_rgb = image.convert("RGB")
               # Perform classification on the uploaded image
               #predictions = import_and_predict(image, model)
               size = (300, 300)
               img = ImageOps.fit(image_rgb, size, Image.ANTIALIAS)
                # Convert Image to a numpy array
               img_array = np.array(img, dtype=np.uint8)
                # Scaling the Image Array values between 0 and 1
               img_array =  img_array/255.0
                
               img_reshape = img_array[np.newaxis, ...]
                # Get the Predicted Label for the loaded Image
               p =model.predict(img_reshape)
               #class_names = ['Bees', 'Butterfly','Beetles',  'Cicada','Dragonfly','Grasshopper', 'Ladybird','Mosquito', 'Moth', 'Scorpion','Snail','Spider']
               #score = tf.nn.softmax(predictions[0])
               #predicted_class = class_names[np.argmax(score)]
               #confidence = 100 * np.max(score)
               labels = {
               0: 'Bees',
               1: 'Beetles',
               2: 'Butterfly',
               3: 'Cicada',
               4: 'Dragonfly',
               5: 'Grasshopper',
               6: 'Ladybird',
               7: 'Mosquito',
               8: 'Moth',
               9: 'Scorpion',
               10: 'Snail',
               11: 'Spider'
           }
               print("\n\nMaximum Probability: ", np.max(p[0], axis=-1))
               predicted_class = labels[np.argmax(p[0], axis=-1)]
               print("Classified:", predicted_class, "\n\n") 

               # Display the classification result
               #st.success("This image most likely belongs to {}".format(predicted_class, confidence))
               string =(
               "This image most likely belongs to {}."
               .format(predicted_class)
               )
               st.success(string)
               #####################firebase upload
               user = auth.sign_in_with_email_and_password(st.session_state.email,st.session_state.password)
               uid = user['localId']
               fireb_upload = storage.child(file.name).put(file.getvalue(),user['idToken'])
               a_imgdata_url = storage.child(file.name).get_url(fireb_upload['downloadTokens']) 
                   
    
               now = datetime.now()
               dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
               post = {'Farm_ID' :farmid,
                       'Camera_ID' : cameraid,
                       'Classification' : string,
                       'Time Stamp' : dt_string,
                       'Image URL' : a_imgdata_url}                           
               results = db.child(user['localId']).child("classification").push(post)
                   
               st.success('Saved in firebase')
############################upload
   elif bio == 'Upload':
       st.write('Dataset upload Page')
       img = st.file_uploader('Upload an image',type=["jpg", "png"])
       name = st.text_input("Bird/Insect/weevil Name",max_chars = 100)
       upload=st.button('Upload')
       if upload and name is not None and img is not None :
           #saving in firebase
           user = auth.sign_in_with_email_and_password(st.session_state.email,st.session_state.password)
           uid = user['localId']
           fireb_upload = storage.child(img.name).put(img.getvalue(),user['idToken'])
           a_imgdata_url = storage.child(img.name).get_url(fireb_upload['downloadTokens']) 
           
           
           now = datetime.now()
           dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
           post = {'pestname' : name,
                   'Timestamp' : dt_string,
                   'imgURL' : a_imgdata_url}                           
           results = db.child(user['localId']).child("Donations").push(post)
           
           st.success('Thank you for your upload')     
       else:
           st.warning('Please provide all the information above')
   


def home():
    st.title("Welcome to IPests!")
    # Add your content for the home page here
    st.write("Please enter your choise.")
    
    choice = st.selectbox('login/Signup', ['Login', 'Sign up'])

    # Username input
    #username = st.text_input("Username")

    # Password input
    #password = st.text_input("Password", type="password")
    
    #signup
    
    if choice == 'Sign up':
        handle = st.text_input(
            'Please input your app handle name', value='Default')
        
        # Obtain User Input for email and password
        email = st.text_input('Please enter your email address')
        password = st.text_input('Please enter your password',type = 'password')
        submit = st.button('Create my account')
        if submit:
            #auth.send_email_verification()
          
            auth.create_user_with_email_and_password(email, password)
            user = auth.sign_in_with_email_and_password(email, password)
            auth.send_email_verification(user["idToken"])

            st.success('Your account is created suceesfully!')
            #st.balloons()s
            # Sign in
            #user = auth.sign_in_with_email_and_password(email, password)
            #auth.send_email_verification(email)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("ID").set(user['localId'])
            st.title('Welcome ' + handle)
            st.info('Login via login drop down selection')
########################################################################
    if choice == 'Login':
        st.session_state.email = st.text_input('Please enter your email address')
        st.session_state.password = st.text_input('Please enter your password',type = 'password')
        login = st.button('Login')
        if login:
            user = auth.sign_in_with_email_and_password(st.session_state.email,st.session_state.password)
            
            user_obj = auth.get_account_info(user["idToken"])
            email_verified = user_obj["users"][0]["emailVerified"]
            if email_verified:
                st.success("Logged in successfully!")
                # Set session state to indicate successful login
                st.session_state.logged_in = True
                st.experimental_rerun()
            
               
               

            else:
                st.error("Please verify your e-mail address.")           


if __name__ == '__main__':
    
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        
        home()
    else:
        main()
