import tkinter as tk
from tkinter import filedialog
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model architecture from JSON file
json_file_path = 'C:\\Users\\user\\Desktop\\Sign Language detection\\asl_model.json'  # Adjust based on your directory
weights_file_path = 'C:\\Users\\user\\Desktop\\Sign Language detection\\asl_model_weights.weights.h5'
# Load the model architecture
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the model weights
model.load_weights(weights_file_path)

# Dictionary for labels (customize based on your dataset)
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
                8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
                15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 
                22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Tkinter window setup
window = tk.Tk()
window.title("Sign Language Detection")
window.geometry("800x600")

# Label to display uploaded image or video feed
label_display = Label(window)
label_display.pack()

# Variable to control video capture
video_running = False
cap = None

# Function to upload and predict from image
def upload_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename()
    
    # Read and preprocess the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (grayscale)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    
    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    # Display the result
    result_label.config(text=f"Predicted Sign: {class_labels[predicted_class]}")
    
    # Display the image in the GUI
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    label_display.config(image=img_tk)
    label_display.image = img_tk

# Function for real-time video prediction
def start_real_time_video():
    global video_running, cap
    if video_running:
        return  # Exit if already running
    
    video_running = True
    cap = cv2.VideoCapture(0)
    
    def update_frame():
        global video_running, cap
        if not video_running:
            cap.release()
            return
        
        ret, frame = cap.read()
        if not ret:
            video_running = False
            cap.release()
            return
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (48, 48))
        input_frame = np.expand_dims(resized_frame, axis=-1)  # Add channel dimension
        input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
        input_frame = input_frame / 255.0  # Normalize
        
        # Predict the sign
        prediction = model.predict(input_frame)
        predicted_class = np.argmax(prediction)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Sign: {class_labels[predicted_class]}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Convert the frame to RGB for Tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update the label to display the video feed
        label_display.config(image=img_tk)
        label_display.image = img_tk
        
        # Schedule the next frame update
        window.after(10, update_frame)
    
    update_frame()

def stop_real_time_video():
    global video_running, cap
    video_running = False
    if cap is not None:
        cap.release()

# Buttons for GUI actions
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

video_start_button = tk.Button(window, text="Start Real-Time Video", command=start_real_time_video)
video_start_button.pack()

video_stop_button = tk.Button(window, text="Stop Real-Time Video", command=stop_real_time_video)
video_stop_button.pack()

# Label to display prediction results
result_label = tk.Label(window, text="Predicted Sign: None", font=('Arial', 16))
result_label.pack()

# Start the GUI
window.mainloop()
