# gui/gui_app.py
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_sample(model, sample):
    # Scale the sample
    scaler = StandardScaler()
    sample = scaler.fit_transform(np.array([sample]))
    
    # Predict
    prediction = model.predict(sample)
    return prediction.argmax(axis=1)[0]

def classify():
    try:
        model = tf.keras.models.load_model('models/xlstm_model.h5')
        sample = [float(x) for x in entry.get().split(',')]
        prediction = predict_sample(model, sample)
        messagebox.showinfo("Prediction", f"The predicted class is: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI
root = tk.Tk()
root.title("Intrusion Detection System")

entry = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Enter feature vector (comma-separated):")
label.pack()

input_field = tk.Entry(frame, textvariable=entry, width=50)
input_field.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

classify_button = tk.Button(button_frame, text="Classify", command=classify)
classify_button.pack()

root.mainloop()
