import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Load the model and weights
def FacialExpressionModel(json_file, weights_file):
    try:
        with open(json_file, 'r') as file:
            loaded_model_json = file.read()
            model = model_from_json(loaded_model_json)
        
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model and weights loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model or weights: {e}")
        return None

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

# Labels and buttons
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = FacialExpressionModel('model_a1.json', 'model_weights1.h5')

EMOTIONS_LIST = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Detect emotion in the image
def Detect(file_path):
    global label1
    try:
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Unable to load image.")
            label1.configure(foreground='#011638', text="Error: Unable to load image")
            return
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            print("No face found in the image.")
            label1.configure(foreground='#011638', text="No face found")
            return
        
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        print(f"Predicted emotion: {pred}")
        label1.configure(foreground='#011638', text=pred)
    except Exception as e:
        print(f"Error during detection: {e}")
        label1.configure(foreground='#011638', text="Error detecting emotion")

# Show the "Detect" button after uploading an image
def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

# Upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.3), (top.winfo_height() / 2.3)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

# GUI layout
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'))
upload.pack(side=tk.BOTTOM, pady=50)
sign_image.pack(side=tk.BOTTOM, expand=True)
label1.pack(side=tk.BOTTOM, expand=True)
heading = Label(top, text="Emotion Detector", pady=20, font=('arial', 24, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

# Run the GUI
top.mainloop()