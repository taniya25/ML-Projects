#Step 1 = read and show video stream, capture image
#Step 2 = detect faces and show bounding box
#step 3 = Flatten the largest face image(grayscale image) and store as a numpy array
#Step 4 = repeat the above for multiple people and generate training data

import cv2
import numpy as np

cap = cv2.VideoCapture()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
face_data = [] #we will store the data(coordinates) of every face in this list
data_path = './Datasets/'

#Input the name of the person for whom you're creating this dataset, it will later be the name of the file in which we'll save the training data for that person
file_name = input("Enter name of the Person : ")

while True:

	ret, frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	gray_frame = cv2.cvtColor(frame,cvt.COLOR_BGR2GRAY)

	#print all the frames 
	print(faces)#This will give us a list of coordinates where the face is detected
	#After we get all the faces, we sort the face data according to area
	#So that the largest face gets selected as required
	#So we'll calculate the area by multiplying the height and width tuples in the list
	#So faces[2]*faces[3] is the key around which we need to sort our data, we will use lamba to do that 

	faces = sorted(faces,key=lambda f:f[2]*f[3],reverse = True)

	#Pick the first face as that would be the largest
	for x,y,w,h in faces[0:]:
		cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(250,0,0),2)

		#Extract(crop out the required region) : region of interest
		offset = 10 #this will determine the pixel padding to be given while cropping\
		face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
		#we'll slice the frame list and as y axis comes before x axis by convention we follow above scheme
		face_section = cv2.resize(face_section,(100,100))
		skip += 1

		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Frame",gray_frame)
	cv2.imshow("face",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

#convert our face list into a numpy array
face_data = npasarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data)

#Save this data into file system 	
np.save(data_path+file_name+'.npy',face_data)

print("Data Successfully saved at "+data_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()