import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dis
#import espeak
import time
import os
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

def draw_landmarks(image, outputs, land_mark, color):
    height, width =image.shape[:2]
             
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        cv2.circle(image, point_scale, 2, color, 1)
        
def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    
    distance = dis.euclidean(point1, point2)
    return distance



def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis/ top_bottom_dis
    
    return aspect_ratio




mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


STATIC_IMAGE = False
MAX_NO_FACES = 1
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)

LIPS=[ 61, 146, 91, 181, 84, 17, 31+4, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]


FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]


cap = cv2.VideoCapture(0)

frame_count = 0
yawn_count = 0
min_frame = 5
min_tolerance = 4.5

#speech = pyttsx3.init()

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
   
    if results.multi_face_landmarks:

           
        draw_landmarks(image, results, FACE, COLOR_GREEN)
        draw_landmarks(image, results, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
        draw_landmarks(image, results, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
            
        ratio_left =  get_aspect_ratio(image, results, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            
    
        draw_landmarks(image, results, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
        draw_landmarks(image, results, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
            
        ratio_right =  get_aspect_ratio(image, results, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            
        ratio = (ratio_left + ratio_right)/2.0
            
            
        if ratio > min_tolerance:
            frame_count +=1
        else:
            frame_count = 0
            
        cv2.putText(image,"STATUS: ", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image,"EAR: " +  str(np.round(ratio,2)), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image,"EAR Counter: " +  str(np.round(frame_count,2)), (395, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if frame_count > min_frame:
            #Closing the eyes
            cv2.putText(image, "Long Eye Closure", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            #speech.say('Drowsy Alert: It Seems you are dozzing off.. please wake up!')
            #speech.runAndWait()
                
        draw_landmarks(image, results, UPPER_LOWER_LIPS , COLOR_BLUE)
        draw_landmarks(image, results, LEFT_RIGHT_LIPS, COLOR_BLUE)
            
        ratio_lips =  get_aspect_ratio(image, results, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
        
        if ratio_lips < 1.8:
            yawn_count +=1
        else:
            yawn_count = 0
        
        cv2.putText(image,"MAR Counter: " +  str(np.round(yawn_count,2)), (395, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if ratio_lips < 1.8:
            #Open his mouth
            cv2.putText(image, "Yawning", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            #speech.say('Drowsy Warning: You look tired.. please take a rest')
            #speech.runAndWait()
                
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
            
                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
                    
                    
                    
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          


            # See where the user's head tilting
            if y < -1.25 and yawn_count >= 4 or y < -1.25 and frame_count >= 8:
                text = "Moderately Drowsy"
                
                #color1()
            elif x < -3.85:
                text = "Extremely Drowsy"
                
                #color1()
            elif y > -1.15 and yawn_count >= 4 or y > -1.15 and frame_count >= 8:
                text = "Moderately Drowsy"
                
            else:
                text = "Normal"
                


            # Add the text on the image
            cv2.putText(image, "MAR: " + str(np.round(ratio_lips,2)), (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Rotation Degree: ", (395, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (155, 225, 255), 2)
            cv2.putText(image, text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (470, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (155, 225, 225), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (470, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (155, 225, 225), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (470, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (155, 255, 225), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
    

    cv2.imshow('Head Bending', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()


