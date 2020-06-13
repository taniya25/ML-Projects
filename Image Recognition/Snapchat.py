import numpy as np
import cv2



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

cap = cv2.VideoCapture(0)
moustache = cv2.imread('images/mustache.png',-1)
glasses = cv2.imread('images/glasses.png',-1)
face_cascade = cv2.CascadeClassifier('xml_files/haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('xml_files/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('xml_files/Nose18x15.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray            = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces           = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if ret==False:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray    = gray[y:y+h, x:x+h] # rec
        roi_color   = frame[y:y+h, x:x+h]
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = image_resize(glasses.copy(), height = 15*eh,width=ew + 100)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    #print(glasses[i, j]) #RGBA
                    if glasses2[i, j][3] != 0: # alpha 0
                        roi_color[ey + i  , ex + j ] = glasses2[i, j]


        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            moustache2 = image_resize(moustache.copy(), height = 5*nh,width=nw + 10)

            mw, mh, mc = moustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    #print(glasses[i, j]) #RGBA
                    if moustache2[i, j][3] != 0: # alpha 0
                        roi_color[ny + int(nh/1.5) + i, nx + j] = moustache2[i, j]

    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()