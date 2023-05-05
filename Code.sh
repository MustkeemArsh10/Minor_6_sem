########code for path tracing##########

import numpy as np
import cv2
from time import sleep

cap = cv2.VideoCapture(0)

# initialize previous laser point coordinates
prev_x, prev_y = None, None

# create empty black image for drawing the laser path
path_img = np.zeros((480, 640, 3), dtype=np.uint8)

while(cap.isOpened()):
    ret, frame = cap.read()

    # splits frame in 3 color channels
    b, g, r = cv2.split(frame)

    # creates color range
    minRed = np.array(254)
    maxRed = np.array(255)

    # applies color range mask
    maskRed = cv2.inRange(r, minRed, maxRed)
    resultRed = cv2.bitwise_and(r, r, mask=maskRed)

    # creates kernel, then erode and then dilates
    kernel = np.ones((3, 3), np.uint8)
    resultRed = cv2.erode(resultRed, kernel, iterations=1)
    resultRed = cv2.dilate(resultRed, kernel, iterations=2)

    ############# PUTS COORDS OF LASER POINT ON SCREEN FRAME #############################

    # calculate moments of binary image
    M = cv2.moments(resultRed)

    if (M["m00"] != 0):
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        coord = '(' + str(cX) + ' | ' + str(cY) + ')'

        # put text on image
        cv2.putText(frame, "laser", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, coord, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # draw line from previous to current laser point on path_img if prev_x and prev_y are not None
        if prev_x is not None and prev_y is not None:
            cv2.line(path_img, (prev_x, prev_y), (cX, cY), (0, 255, 0), 2)

        # update previous laser point coordinates
        prev_x, prev_y = cX, cY

    else:
        cv2.putText(frame, "NOT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    ######################################################################################

    ############# SHOWS SCREEN FRAME #####################################################

    # add path_img to frame and show
    frame_with_path = cv2.addWeighted(frame, 0.8, path_img, 0.2, 0)
    resized = cv2.resize(frame_with_path, (800, 400), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(resultRed, (800, 400), interpolation=cv2.INTER_AREA)

    cv2.imshow('Realtime track',resized)
    cv2.imshow('Threshold',resized2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ######################################################################################

cap.release()
cv2.destroyAllWindows()




############## Code for Coordinates ################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import sleep
import math




# calculate distance between two coordinates
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)*2 + (y2 - y1)*2)

#cordinates 

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
p = frame_height/2
q = frame_width/2

count = 0
prev_count = 0
chount = 0 # for countint shots that means count of laser stop lighting
x = np.array(0) # will store distance between center and co-ordinates of laser
y = np.array(0) # store no. of shots or count of shots
cX = 0 # x- coordinate of laser
cY = 0 # y- coordinate of laser





while(cap.isOpened()):
    ret, frame = cap.read()

    # splits frame in 3 color channels
    b,g,r = cv2.split(frame)

    # creates color range
    minRed = np.array(254)
    maxRed = np.array(255)

    # applies color range mask

    #The inRange function takes three input parameters: the source image, 
    # a lower boundary value, and an upper boundary value. It then returns
    #  a binary (black-and-white) image where white pixels represent the pixels
    #  in the source image that fall within the specified range, and black pixels
    #  represent those that do not.
    maskRed = cv2.inRange(r, minRed, maxRed)

    

    resultRed = cv2.bitwise_and(r, r, mask = maskRed)

    # creates kernel, then erode and then dilates
    kernel = np.ones((3, 3), np.uint8)
    resultRed = cv2.erode(resultRed, kernel, iterations = 1)
    resultRed = cv2.dilate(resultRed, kernel, iterations = 2) 


    ############# PUTS COORDS OF LASER POINT ON SCREEN FRAME #############################
    
    # calculate moments of binary image
    M = cv2.moments(resultRed)

    if (M["m00"] != 0):
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        #print("Laser coordinates: ({}, {})".format(cX, cY))
        #333333333333333
        count += 1
        # y = np.append(y,count)
        #x = np.append(x,distance(cX,p,cY,q))

        #33333333333333333
    
        coord = '(' + str(cX) + ' | ' + str(cY) + ')'
        # put text on image
        cv2.putText(frame, "shot", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, coord, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "NOT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if count>1 and count > prev_count:
            x = np.append(x,distance(q,p,cX,cY))
          ##  print(q,p,cX,cY,distance(q,p,cX,cY))
            prev_count = count
            chount += 1
            y = np.append(y,chount)
    ######################################################################################

    ############# SHOWS SCREEN FRAME #####################################################

    #resized =  cv2.resize(frame, (600, 600), interpolation = cv2.INTER_AREA)
    #resized2 =  cv2.resize(resultRed, (600, 600), interpolation = cv2.INTER_AREA)
    cv2.imshow('frame',frame) #resized
    cv2.imshow('lala',resultRed) #resized2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ######################################################################################
    

cap.release()
cv2.destroyAllWindows()



##3333333333333333333333333333
fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_xlabel('Distance of Shot from Target(in px)')
ax.set_ylabel('Registered Shots')
plt.show()

print(p,q)
print(x)
