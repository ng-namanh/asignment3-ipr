import tkinter as tk
from human_detection_image import open_file_dialog

def close_application():
    root.destroy() 

root = tk.Tk()
root.title("Human Detection Application")

label = tk.Label(root, text="Human Detection Application")
label.pack()


upload_button = tk.Button(root, text="Upload Image", command=open_file_dialog)
upload_button.pack(pady=20)

close_button = tk.Button(root, text="Close", command=close_application)
close_button.pack(pady=20)

root.mainloop() 