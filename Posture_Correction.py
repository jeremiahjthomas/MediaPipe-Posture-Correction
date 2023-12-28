import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#calculates angle between three points, a, b, c
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle

#calculates the angle between two points
def calculate_angle2(a, b):
    c = np.array([a[0], b[1]])
    return calculate_angle(b, a, c)

#calculates midpoint between two points
def calculate_midpoint(a, b):
    return (a[0]+b[0])/2 , (a[1]+b[1])/2

#calculates distance between two points
def calculate_distance(a, b):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

#The code below chooses the correct source
cap = cv2.VideoCapture(0)
try:
    imageTest = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
except:
    cap = cv2.VideoCapture(1)

#sets the width to 1000 and height to 600
cap.set(3, 1000)
cap.set(4, 600)

#counts frames to test crash speed
frame_counter = 0

#set up mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #set up opencv instance
    while cap.isOpened():
        #read in ret (if returned anything) and frame and adds to counter
        ret, frame = cap.read()
        frame_counter+=1

        # Detect pose and render
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor frame from bgr to rgb because open cv is default bgr 
        frame.flags.writeable = False #saving memory
        
        # Making detection
        results = pose.process(frame) #array
        try:
            results2 = mp_face_mesh.FaceMesh(refine_landmarks=True).process(frame)
        except:
            results2 = None

        # Recolor 
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #get landmark coordinates
        try:
            landmarks = results.pose_landmarks.landmark
            
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]  
            leftShoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y] 
            rightShoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y] 
            centerShoulder = calculate_midpoint(leftShoulder, rightShoulder)
            elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y] 
            wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y] 
            leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            rightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            leftEar =  [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            rightEar =  [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            mouth =  [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            pelvis = calculate_midpoint(leftHip, rightHip)
            leftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            rightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            ptBetweenAnkles = calculate_midpoint(leftAnkle, rightAnkle)
        #if it isnt seen dont throw an error
        except:
            pass
        
        #for facemesh landmarks use global vars
        chin = [0,0]
        tip_of_head = [0,0]
        try:
            landmarks2 = results2.multi_face_landmarks[0].landmark
            chin = [landmarks2[152].x, landmarks2[152].y]
            tip_of_head = [landmarks2[10].x, landmarks2[10].y]
        except:
            pass
    
        #calculate posture markers
        trunkAngle = calculate_angle(leftShoulder, leftHip, knee)
        headAngle = calculate_angle2(leftEar, mouth)
        straightness = calculate_angle(tip_of_head, centerShoulder, pelvis)
        
        #output marker text
        color1 = (255, 255, 255) if 174 < trunkAngle < 180 else (0, 0, 255)
        cv2.putText(frame, f"Trunk Angle: {str(trunkAngle)}", (70,60), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2, cv2.LINE_AA)

        color2 = (255, 255, 255) if headAngle > 10 else (0, 0, 255)
        cv2.putText(frame, f"Head Angle: {str(headAngle)}", (70,100), cv2.FONT_HERSHEY_SIMPLEX, 1,color2, 2, cv2.LINE_AA)
        
        color3 = (255, 255, 255) if calculate_distance(chin, centerShoulder)*100 > 13 else (0, 0, 255)
        cv2.putText(frame, f"Relaxed Shoulders: {str(calculate_distance(chin, centerShoulder)*100)}%",
                            (70,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color3, 2, cv2.LINE_AA)
        color4 = (255, 255, 255) if 150 < straightness < 180  else (0, 0, 255)
        cv2.putText(frame, f"Straightness: {str(straightness)}", (70,180), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2, cv2.LINE_AA)

        #draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,60), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

        #show frame
        cv2.imshow('Looks', frame)

        #quit on q-key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        #avoid memory loss/leakage
        del frame 

#avoid memory loss/leakage and check crash speed
cap.release()
cv2.destroyAllWindows()
print(frame_counter)

