import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile

root = tk.Tk()

canvas = tk.Canvas(root, width=600, height=300)
canvas.grid(columnspan=3, rowspan=3)

#logo
logo = Image.open('logo.png')
logo = ImageTk.PhotoImage(logo)

logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=1, row=0)

#instruction
instructions = tk.Label(root, text="Select an Image file on your computer to classify", font="Raleway")
instructions.grid(columnspan=3, column=0, row=1)

def open_file():
	print("is this working ?")
	browse_text.set("loading...")
	file = askopenfile(parent=root, mode='rb', title="Choose an image", filetype=[("JPG", "*.jpg", "PNG", "*.png")])
	if file:
		print("file was successfuly loaded")

#browse button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:open_file(), font="Raleway", bg='#20bebe', fg='white', height=2, width=15)
browse_text.set("Browse")
browse_btn.grid(column=1, row=2)

root.mainloop()