import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from main import detectPose
from predefined_angles.parvatasana import parvatasana
from predefined_angles.pranamasana import pranamasana
from predefined_angles.Hasta_Uttanasana import HastaUttasana
from predefined_angles.Ashtanga_Namaskara import AshtangaNamaskara
from predefined_angles.Bhujangasana import Bhujangasana
from predefined_angles.Hasta_padasana import HastaPadasana
from predefined_angles.Dhandasana import Dhandasana
# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose()

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle



def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    # Specify the color (Red) with which the label will be written on the image.
    color = (255, 0, 0)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    
    left_hip = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    right_hip = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    
    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
     
    right_shoulder = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    
    left_shoulder = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
  


    print(f'left_elbow_angle: {left_elbow_angle}\n right_elbow_angle:{right_elbow_angle}\
          left_knee_angle:{left_knee_angle}\n right_knee_angle:{right_knee_angle}\
          left_hip: {left_hip}\n right_hip : {right_hip}\n left_should:{left_shoulder}\n\
            right shoulder: {right_shoulder}')
    feedback = Dhandasana(left_elbow=left_elbow_angle, 
                      right_elbow=right_elbow_angle,
                      left_hip= left_hip,
                      right_hip= right_hip,
                      left_knee=left_knee_angle,
                      right_knee=right_knee_angle)
    

    cv2.putText(output_image,f'left elb:{left_elbow_angle}, right_elb:{right_elbow_angle}', (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(output_image, f'left hip:{left_hip}, right hip:{right_hip}', (10, 60),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(output_image, f'left shoul:{left_shoulder}, right shoul:{right_shoulder}', (10, 90),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    cv2.putText(output_image, feedback, (10, 120),cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        cv2.imwrite('output_image.jpg', output_image)
        
    else:
        
        # Return the output image and the classified label.
        return output_image, feedback
    


# image = cv2.imread('poses/Dhandasana.jpg')
# output_image, landmarks = detectPose(image, pose, display=False)
# if landmarks: 
#     _, feedback = classifyPose(landmarks, output_image)
#     print(feedback)




# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    # example_img = cv2.imread(f'poses/parvatasana.jpg')
    # size = 200
    # example_img = cv2.resize(example_img, (size,size))

    # img2gray = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY) 
    # ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY) 
    
    # Read a frame.
    ok, frame = camera_video.read()

    # roi = frame[-size-270:-270, -size-420:-420] 

    # # Set an index of where the mask is 
    # roi[np.where(mask)] = 0
    # roi += example_img 
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame and ignore the empty camera frame.
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose, display=False)
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        frame, feedback = classifyPose(landmarks, frame, display=False)
    
    # Display the frame.
    cv2.imshow('Pose Classification', frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) & 0xFF
    
    # Check if 'ESC' is pressed.
    if(k == 27):
        
        # Break the loop.
        break

# Release the VideoCapture object and close the windows.
camera_video.release()
cv2.destroyAllWindows()