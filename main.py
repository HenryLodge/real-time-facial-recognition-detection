from tkinter import *
from tkinter import filedialog
import cv2 as cv
import sys
from PIL import Image, ImageTk
import numpy as np
from PIL import Image, ImageTk
import os
import datetime
from socket import *
import time as t


root2 = Tk()
root2.title('Info')
root2.geometry('1000x800')
root2.configure(bg='gray14')
root2.resizable(0, 0)


def take_picture():
	name_gotten2 = name_box.get()
	ran = 0
	image1 = Image.fromarray(img1_2)
	time = str(datetime.datetime.now().today()).replace(':',' ') + '.jpg'
	new_path = f'C:/OCV/real-time-FR/pictures/{name_gotten2}/'
	os.mkdir(new_path)
	image1.save(time)
	time2 = f'{new_path}{time}'
	os.rename(time, time2)
	ran = 1
	print(ran)
	if ran == 1:
		image2 = Image.fromarray(img1_2)
		time3 = str(datetime.datetime.now().today()).replace(':',' ') + '.jpg'
		time4 = f'{new_path}{time3}'
		image2.save(time4)
		ran = 2
		if ran == 2:
			image3 = Image.fromarray(img1_2)
			time4 = str(datetime.datetime.now().today()).replace(':',' ') + '.jpg'
			time5 = f'{new_path}{time4}'
			image2.save(time5)
			ran = 3
			print('Done taking pictrues')


def train_model():
	global name_gotten3
	name_gotten3 = name_box.get()
	people = [f'{name_gotten3}']
	DIR = r'C:/OCV/real-time-FR/pictures'

	haar_cascade = cv.CascadeClassifier('haar_face.xml')

	features = []
	labels = []


	def create_train():
		for person in people:
			path = os.path.join(DIR, person)
			label = people.index(person)

			for img in os.listdir(path):
				img_path = os.path.join(path,img)

				img_array = cv.imread(img_path)
				gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

				faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12)

				for (x,y,w,h) in faces_rect:
					faces_roi = gray[y:y+h, x:x+w]
					features.append(faces_roi)
					labels.append(label)

	create_train()

	features = np.array(features, dtype='uint8')
	labels = np.array(labels)

	face_recognizer = cv.face.LBPHFaceRecognizer_create()
	print(type(features))
	print(type(labels))

	# training on features and labels list
	face_recognizer.train(features, labels)

	face_recognizer.write('face-trainedreal.yml')

	np.save('featuresreal.npy', features)

	np.save('labelsreal.npy', labels)

	print('Done training the model')


def next_window():
	root2.destroy()

	root1 = Tk()
	root1.title('Real-Time Facial Detection/Recognition')
	root1.geometry('1400x700')
	root1.configure(bg='gray14')
	root1.resizable(0, 0)
	global img
	global test123

	# button functions
	def detect_on():
		global img
		global test123
		test123 = 2
		while test123 == 2:
			haar_cascade = cv.CascadeClassifier('haar_face.xml')
			img = cam.read()[1]
			faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)

			for (x,y,w,h) in faces_rect:
				img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

			img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
			img = ImageTk.PhotoImage(Image.fromarray(img))
			L1['image'] = img
			root1.update()

	def detect_off():
		global img
		global test123
		test123 = 1
		img = cam.read()[1]
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = ImageTk.PhotoImage(Image.fromarray(img))
		L1['image'] = img
		root1.update()

	def recog_on():
		global img
		global test123
		global name_gotten3
		test123 = 3
		while test123 == 3:
			haar_cascade = cv.CascadeClassifier('haar_face.xml')

			people = [{name_gotten3}]

			face_recognizer = cv.face.LBPHFaceRecognizer_create()
			face_recognizer.read('face-trainedreal.yml')

			img = cam.read()[1]
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

			# detect the face
			if True:
				faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

				for (x,y,w,h) in faces_rect:
					faces_roi = gray[y:y+h, x:x+w]

					label, confidence = face_recognizer.predict(faces_roi)
					print(f'Label = {people[label]} with a confidence of {confidence}')

					cv.putText(img, str(people[label]), (x+2,y+h-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), thickness=2)
					cv.rectangle(img, (x-5,y-5), (x+w,y+h),(0,255,0), thickness=2)

				img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
				img = ImageTk.PhotoImage(Image.fromarray(img))
				L1['image'] = img
				root1.update()

	def recog_off():
		global img
		global test123
		test123 = 1
		img = cam.read()[1]
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = ImageTk.PhotoImage(Image.fromarray(img))
		L1['image'] = img
		root1.update()

	def name_get():
		global name_gotten
		name_gotten = name_box.get()
		print(name_gotten)
		print(type(name_gotten))

	# buttons
	Face_detect_on = Button(root1, text='Start FD', padx=75, pady=25, bg='black', fg='white', font=('Calibri', 30, 'bold'), borderwidth=0, command=detect_on)
	Face_detect_on.place(x=35, y=180)

	Face_detect_off = Button(root1, text='Stop FD', padx=75, pady=25, bg='black', fg='white', font=('Calibri', 30, 'bold'), borderwidth=0, command=detect_off)
	Face_detect_off.place(x=35, y=380)

	# FR buttons
	Face_recog_on = Button(root1, text='Start FR', padx=75, pady=25, bg='black', fg='white', font=('Calibri', 30, 'bold'), borderwidth=0, command=recog_on)
	Face_recog_on.place(x=1060, y=180)

	Face_recog_off = Button(root1, text='Stop FR', padx=75, pady=25, bg='black', fg='white', font=('Calibri', 30, 'bold'), borderwidth=0, command=recog_off)
	Face_recog_off.place(x=1060, y=380)

	Label(root1, text='Camera Output', font=('Calibri', 50, 'bold'), bg='gray14', fg='white').pack()
	f1 = LabelFrame(root1, bg='white')
	f1.pack()
	L1 = Label(f1, bg='white')
	L1.pack()
	test123 = 1
	cam = cv.VideoCapture(0)

	while test123 == 1:
		global img
		img = cam.read()[1]
		img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
		img = ImageTk.PhotoImage(Image.fromarray(img1))
		L1['image'] = img
		root1.update()

	root1.mainloop()


hello_text = Label(root2, text='Enter your name, press Picture, press Train Model, and press Continue', bg='gray14',fg='white', font=('Calibri', 15, 'bold'))
hello_text.place(x=200, y=600)

picture_button = Button(root2, text='Picture', padx=50, pady=25, bg='black', fg='white', font=('Calibri', 15, 'bold'), borderwidth=0, command=take_picture)
picture_button.place(x=300, y=650)

name_box = Entry(root2, bg='black', width=17, fg='white', font=('Calibri', 15, 'bold'), borderwidth=0)
name_box.place(x=100, y=680)

tain_button = Button(root2, text='Train Model', padx=30, pady=25, bg='black', fg='white', font=('Calibri', 15, 'bold'), borderwidth=0, command=train_model)
tain_button.place(x=500, y=650)

continue_button = Button(root2, text='Continue', padx=45, pady=25, bg='black', fg='white', font=('Calibri', 15, 'bold'), borderwidth=0, command=next_window)
continue_button.place(x=700, y=650)

Label(root2, text='Camera Output', font=('Calibri', 50, 'bold'), bg='gray14', fg='white').pack()
f1_2 = LabelFrame(root2, bg='white')
f1_2.pack()
L1_2 = Label(f1_2, bg='white')
L1_2.pack()
cam = cv.VideoCapture(0)

while True:
	img_2 = cam.read()[1]
	img1_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)
	img_2 = ImageTk.PhotoImage(Image.fromarray(img1_2))
	L1_2['image'] = img_2
	root2.update()


root2.mainloop()
