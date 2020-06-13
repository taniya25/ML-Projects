#Classification of faces using the KNN algorithm 
#We want to make a training set and train our algo against it, so that when we input a new face the algo can easily classify it into 
#one of the person. Our algo will actually take the test set and try to recognise whose face it is 

#Read a video stream using OpenCV
#Extract faces out of it
#load training data (numpy of all the persons)
	#x values are stored in numpy 
	#y values we need to assign for each person 
#use KNN to find prediction of face(int)
#map predicted id to name of user
#display the predictions on screen - bounding box and name

import cv2
import numpy as np
import os

def distance(v1,v2):
	#euclidean distance 
	return np.sqrt(sum((v1-v2)**2))

def KNN(train, test, k = 5):
	dist = []
	for i in range(train.shape[0]):
	 	#get vector and label
	 	ix = train[i,:-1]
	 	iy = train[i,-1]
	 	#compute distance from test point 
	 	d = distance(test,ix)
	 	dist.append([d,iy])

	 #sort based on distance and get top k 
	 dk = sorted(dist, key = lambda x:x[0])[:k]
	 #Retrieve only labels 
	 labels = np.array(dk)[:,-1]

	 #Get frequencies of each label 
	 output = np.unique(labels, return_counts = True)
	 #Find max frequency and corresponding label
	 index = np.argmax(output[1])
	 return output[0][index]

#Reading a video stream 
cap = cv2.VideoCapture(0)

#Face Detection using the Haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
dataset_path = './Datasets/'
label = []
face_data = []

class_id = 0 #Labels for the given file 
name = {} #Dictionary used to create mapping between the id and name

#We need to load the training data, so let's do Data Preparation 
for fx in os.listdir(dataset_path): #here we will get all the files in the dataset folder
	if fx.endswith('.npy'):
		name[class_id] = fx[:-4]
		print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)#Filename along with its path, it will load the file
		face_data.append(data_item)

		#Create labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		#For each np file we are computing a numpy array of labels 
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis = 0)

print(face_dataset.shape)
print(face_labels.shape)

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	for x,y,w,h in faces:
		offset = 10
		face_section = gray_frame[y-offset:y+h+offset,w-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		#predict the label using KNN
		out = KNN(trainset,face_section.flatten())

		#Display name of the person on screen along with the bounding box
		pred_name = name[int(out)]
		cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

		cv2.rectangle(gray_fraame,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow("Frame",gray_frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()



