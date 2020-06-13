#Read a video stream from Camera
import cv2 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("./xml_files/haarcascade_frontalface_alt.xml")

while True:
	pass
	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(250,0,0),2)

	cv2.imshow("Video Frame",frame)
	cv2.imshow("Gray Frame", gray_frame)
	#wait for user to input -q then you can stop
	key_pressed = cv2.waitKey(1) & 0xFF
	if(key_pressed == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()



