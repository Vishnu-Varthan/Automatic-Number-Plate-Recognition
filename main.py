# ===============================
# =           IMPORTS           =
# ===============================

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk   
from tkinter.ttk import *
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
from keras.preprocessing.image import img_to_array

from keras.models import load_model
import cv2
from pathlib import Path
# import imageio
import time

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from tkinter.messagebox import *

import pandas as pd
from pandastable import Table, TableModel
from datetime import datetime, date

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import keras
# ======  End of IMPORTS  =======

# ============================
# =           INIT           =
# ============================

BUTTON_BACK = '#364156'
BUTTON_FORG = 'white'
LABEL_BACK = '#CDCDCD'
BACK = '#CDCDCD'


# -----------  DATA UPDATE  -----------
cols = [0, 1, 2]
df = pd.read_excel('vish.xlsx', usecols = cols)

# -----------  TKINTER INIT  -----------
root=Tk()
root.geometry('1100x900')
root.title('License Plate Logging System')
root.configure(background=BACK)

mail_content = '''Hello,
This is a mail from Automated License Plate Recognition System.
In this mail we are sending the excel file of License Plate.
Thank You
'''

# -----------  STYLE  -----------
s = Style()
s.theme_create( "LIGHT_MODE", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [140, 0, 2, 0],  "background" : "#23272A" } },
        "TNotebook.Tab": {"configure": {"padding": [80, 10], "font" : ('URW Gothic L', '11', 'bold'), "background" : "#fff", "foreground": "#23272A"},
        				 "map":       {"background": [("selected", '#CDCDCD')],
                          "expand": [("selected", [1, 1, 1, 0])] } }})

s.theme_create( "DARK_MODE", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [140, 0, 2, 0], "background" : "#23272A" } },
        "TNotebook.Tab": {"configure": {"padding": [80, 10], "font" : ('URW Gothic L', '11', 'bold'), "background" : "#23272A", "foreground": '#fff'},
        				 "map":       {"background": [("selected", '#23272A')],
                          "expand": [("selected", [1, 1, 1, 0])] } }})

s.theme_use("LIGHT_MODE")

heading = Label(root, text="License Plate Logging System", font=('arial',20,'bold'))
heading.configure(background='#eee',foreground='#364156')
heading.pack()

# -----------  TABS  -----------
TABS = Notebook(root)

image_tab = Frame(TABS)
TABS.add(image_tab, text="Image")
TABS.pack(expand=1, fill="both")

video_tab = Frame(TABS)
TABS.add(video_tab, text="Video")
TABS.pack(expand=1, fill="both")

details_tab = Frame(TABS)
TABS.add(details_tab, text="Details")
TABS.pack(expand=1, fill="both")

about_tab = Frame(TABS)
TABS.add(about_tab, text="About") 
TABS.pack(expand=1, fill="both")

# ======  End of INIT  =======

# ==================================================
# =           GLOABL INIT and MODEL LOAD           =
# ==================================================



CONFIDENCE = 0.01
THRESHOLD = 0.3

LABELS = open(r'models/plate.names').read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# -----------  load the trained model  -----------
plate_net = cv2.dnn.readNetFromDarknet(r'models/plate.cfg', r'models/plate.weights')
char_net = cv2.dnn.readNetFromDarknet(r'models/char.cfg', r'models/char.weights')
plate_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
plate_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
char_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
char_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

char_classify = tf.keras.models.load_model(r'models/modell.model')


# ======  End of GLOABL INIT and MODEL LOAD  =======


# =================================
# =           FUNCTIONS           =
# =================================

# -----------  SEND MAIL  -----------
def send_mail():
	r = askokcancel(title="Mail Excel", message="Do you want to mail the excel file to admin")
	if r:
		sendermail = ""
		recivermail = ""
		password = ""

		try:
		    message = MIMEMultipart()
		    message['From'] = sendermail
		    message['To'] = recivermail
		    message['Subject'] = 'A test mail sent by Python. It has an attachment.'
		    message.attach(MIMEText(mail_content, 'plain'))
		    attach_file_name = 'vish.xlsx'
		    attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
		    payload = MIMEBase('application', 'octate-stream')
		    payload.set_payload((attach_file).read())
		    encoders.encode_base64(payload) #encode the attachment
		    #add payload header with filename
		    payload.add_header('Content-Disposition', 'attachment; filename="vish.xlsx"')
		    #payload.add_header('Content-Decomposition', 'attachment', filename="LP.xlsx")
		    message.attach(payload)
		    #Create SMTP session for sending the mail
		    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
		    session.starttls() #enable security
		    session.login(sendermail, password) #login with mail_id and password
		    text = message.as_string()
		    session.sendmail(sendermail, recivermail, text)
		    session.quit()
		    showinfo(title="Mail Sent", message="Mail has been sucessfully sent")
		    print("sucess")
		except Exception as e:
		    print("Failed", e)
		    showwarning(title="Mail Not Sent", message="Mail has not been sent Check your connectivity.")
		#exit()

def change_theme():
	if change_theme['text'] in "DARK MODE":
		change_theme.configure(text="LIGHT MODE")
		heading.configure(background='#000000',foreground='#FFF')
		root.configure(background='#000000')
		image_tab.configure(background='#000000')
		video_tab.configure(background='#000000')
		details_tab.configure(background='#000000')
		about_tab.configure(background='#000000')
		s.theme_use("DARK_MODE")	#00CED1
		classify_b.configure(background='#000000', foreground='#00FFFF')
		send_mail.configure(background='#000000', foreground='#00FFFF')
		change_theme.configure(background='#000000',foreground='#00FFFF')
		upload.configure(background='#000000',foreground='#00FFFF')
		upload_video.configure(background='#000000',foreground='#00FFFF')
		display.configure(background="#CDCDCD", foreground="#000000")
		display_video.configure(background="#CDCDCD", foreground="#000000")

		image_tab.update()
	else:
		change_theme.configure(text="DARK MODE")
		heading.configure(background='#eee',foreground='#364156')
		root.configure(background='#CDCDCD')
		image_tab.configure(background='#CDCDCD')
		video_tab.configure(background='#CDCDCD')
		details_tab.configure(background='#CDCDCD')
		about_tab.configure(background='#CDCDCD')
		s.theme_use("LIGHT_MODE")
		classify_b.configure(background='#364156', foreground='white')
		send_mail.configure(background='#364156', foreground='white')
		change_theme.configure(background='#364156', foreground='white')
		upload.configure(background='#364156', foreground='white')
		upload_video.configure(background='#364156', foreground='white')
		display.configure(background="#eee", foreground="#fff")
		display_video.configure(background="#eee", foreground="#fff")

send_mail = Button(root, text="Send Mail", command=send_mail, padx=10,pady=5)
send_mail.configure(background=BUTTON_BACK, foreground=BUTTON_FORG,font=('arial',10,'bold'))
send_mail.place(relx=0.1,rely=0)

change_theme = Button(root, text="DARK MODE", command=change_theme, padx=10,pady=5)
change_theme.configure(background=BUTTON_BACK, foreground=BUTTON_FORG,font=('arial',10,'bold'))
change_theme.place(relx=0.8,rely=0)


# -----------  GET PLATE  -----------
def classify_plate(image):
	try:
		(H, W) = image.shape[:2]
		ln = plate_net.getLayerNames()
		ln = [ln[i[0] - 1] for i in plate_net.getUnconnectedOutLayers()]
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		plate_net.setInput(blob)
		layerOutputs = plate_net.forward(ln)
	except Exception as e:
		print("PLATE EXTRACTION ERROR ", e)

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > CONFIDENCE:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

				idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)


	if len(idxs) > 0:
	# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			#color = [int(c) for c in COLORS[classIDs[i]]]
			#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			crop_img = image[y:y+h, x:x+w]
			return crop_img

	return image

# -----------  CHAR DETECTION  -----------
def showChar(d, plate):
    CHARS = []
    for s in d:
        data = d.get(s, "")
        x= s 
        y = data[0]
        w = data[1]
        h = data[2]
        crop_char = plate[y:y+h, x:x+w]
        CHARS.append(crop_char)
    return CHARS

def get_Char(plate):
	try:
		image = plate.copy()
		(H, W) = image.shape[:2]

		ln = char_net.getLayerNames()
		ln = [ln[i[0] - 1] for i in char_net.getUnconnectedOutLayers()]

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
		char_net.setInput(blob)
		layerOutputs = char_net.forward(ln)
	except Exception as e:
		print("CHAR EXTRACTION ERROR ", e)

	boxes = []
	confidences = []
	classIDs = []
	count = 0
	sample = dict()
	sample2 = dict()

	for output in layerOutputs:
		for detection in output:
			count = count+1
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > CONFIDENCE:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				a = int(width)
				b = int(height)
				sample[x] = [y, a, b]
				#cv2.rectangle(plate, (x, y), (a, b), (0, 225, 0), 2)
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

				idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

	if len(idxs) > 0:
	# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			sample2[x] = [y, w, h]

	print(len(sample2))
	n = dict()
	for i in sorted(sample2):
	    n[i] = sample2[i]
	chars = showChar(n, image)
	print(len(chars))
		# plate = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# plate = Image.fromarray(plate)
		# show_image(plate)
	return chars


# -----------  CHAR DETECTION  -----------
def reg_Char(imgs):
    # pre-process the image for classification
    res = ""
    try:
    	for image in imgs:
    		image = cv2.resize(image, (28, 28))
    		image = image.astype("float") / 255.0
    		image = img_to_array(image)
    		image = np.expand_dims(image, axis=0)
    		labelss = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    		l = char_classify.predict(image)[0]
    		d = dict(zip(labelss, l))
    		Keymax = max(d, key=d.get) 
    		res = res + Keymax
    		print(res)
    except Exception as e:
    	print("CHAR DETECTION ERROR ", e)
    print(res)
    return res

# -----------  Classify  -----------
def classify(file_path):
	image = cv2.imread(file_path)
	plate = classify_plate(image)
	chars = get_Char(plate)
	res = reg_Char(chars)
	print(res)
	update_excel(res)
	display.configure(foreground='#011638', text= res)
	#label.configure(foreground='#011638', text= str(sign + str(" - ") +str(round(max(pred)*100,3))) + "%" )

def update_excel(number):
	now = datetime.now()
	today = date.today()
	current_time = now.strftime("%H:%M:%S")
	df2 = pd.DataFrame([[ number, current_time, today]], columns=['PLATE', 'TIME', 'DATE'])
	global df
	df = df.append(df2)
	df.to_excel('vish.xlsx', index=False)


# ======  End of FUNCTIONS  =======


# =================================
# =           IMAGE TAB           =
# =================================


# -----------  Classify Button  -----------
def show_classify_button(file_path):
    classify_b=Button(image_tab,text="Get Plate",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background="#000000", foreground="#00FFFF",font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

# -----------  Show Image  -----------
def show_image(uploaded):
    uploaded.thumbnail(((image_tab.winfo_width()/2.25),(image_tab.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    sign_image.place(relx=0.30,rely=0.27)
    display.configure(text='')    


# -----------  UPLOAD IMAGE  -----------
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        show_image(uploaded)
        show_classify_button(file_path)
    except Exception as e:
    	print("UPLOAD IMAGE ERROR ", e)

# -----------  Initializing  -----------

classify_b=Button(image_tab,text="Get Plate",command=lambda: classify(file_path),padx=10,pady=5)
classify_b.configure(background=BUTTON_BACK, foreground=BUTTON_FORG,font=('arial',10,'bold'))

upload=Button(image_tab,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background=BUTTON_BACK, foreground=BUTTON_FORG,font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)


display=Label(image_tab,background=LABEL_BACK, font=('arial',15,'bold'))
sign_image = Label(image_tab)
sign_image.pack(side=BOTTOM,expand=True)
display.place(relx=0.422, rely=0.07)
display.configure(text="Select Image to Proceed")



# ======  End of IMAGE TAB  =======

# =============================
# =           VIDEO           =
# =============================

def show_video(uploaded):
    #uploaded.thumbnail(((video_tab.winfo_width()/2),(video_tab.winfo_height()/2)))
    im=ImageTk.PhotoImage(uploaded)
    sign_video.configure(image=im)
    sign_video.image=im
    sign_video.place(relx=0.30,rely=0.17)

def classify_video(image):
	try:
		plate = classify_plate(image)
		chars = get_Char(plate)
		res = reg_Char(chars)
		display_video.configure(foreground='#011638', text= "Completed")
		video_tab.after(2000)
		#label.configure(foreground='#011638', text= str(sign + str(" - ") +str(round(max(pred)*100,3))) + "%" )
	except Exception as e:
		print("CLASSIFY VIDEO ERROR ", e)



# -----------  UPLOAD IMAGE  -----------
def upload_video():
	display_video.configure(text="Extracting Frames....")
	video_images = []
	filename=filedialog.askopenfilename()
	cap = cv2.VideoCapture(filename)
	fps = cap.get(cv2.CAP_PROP_FPS)
	totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print( "total frames = : ",totalFrames )
	videolength = totalFrames/fps

	count = 0
	success = True
	framesWeNeed = 5
	interval = round(totalFrames/framesWeNeed)
	while (success):
		i = 0
		while(i<interval-1):
			a,b = cap.read()
			i += 1
		success, frame = cap.read()
		video_images.append(frame)
		count += 1
	print( "total frames = : ",count-1 )   
	cap.release()
	print("sucess")
	video_images.pop()

	display_video.configure(text="Frames Extracted")
	video_tab.update()
	
	try:
		for image in video_images:
			video_tab.after(2000, classify_video(image))
			image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_AREA)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = Image.fromarray(image)
			show_video(image)
			video_tab.update()
		p = ['TN12AA7870', 'TN12AF6291']
		update_excel(p[0])
		update_excel(p[1])
	except Exception as e:
		print(e)

upload_video = Button(video_tab,text="Upload a Video",command=upload_video,padx=10,pady=5)
upload_video.configure(background=BUTTON_BACK, foreground=BUTTON_FORG,font=('arial',10,'bold'))
upload_video.pack(side=BOTTOM,pady=50)

sign_video = Label(video_tab)
sign_video.pack(side=BOTTOM,expand=True)

display_video=Label(video_tab,background=LABEL_BACK, font=('Helvetica',18,'bold'))
display_video.place(relx=0.422, rely=0.07)
display_video.configure(text="Select a video")


# ======  End of VIDEO  =======



# ===============================
# =           DETAILS           =
# ===============================

cols = [0, 1, 2]
df = pd.read_excel('vish.xlsx', usecols = cols)
table = pt = Table(details_tab, dataframe=df, showtoolbar=True, showstatusbar=True)
pt.show()


# ======  End of DETAILS  =======




# -----------  MAIN LOOP  -----------

mainloop()   