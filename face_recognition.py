# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np


# Import RPi packege provides a class to control the GPIO
import RPi.GPIO as GPIO

# work whit dates and hours
from time import sleep

#import class ThreadPoolExecutor from futures
from concurrent.futures import ThreadPoolExecutor

# use the number of pins
GPIO.setmode(GPIO.BOARD)

 
# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.createLBPHFaceRecognizer()

# Load the trained mode
recognizer.load('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)


# Initialize estate of servomotor
servo_Uno = 11

GPIO.setup(servo_Uno, GPIO.OUT)



estadoServo = 0
# Function move servomotor
def moveServo():
    global estadoServo
    pulso = GPIO.PWM(servo_Uno, 50)
    pulso.start(2.5)	
    for i in range(0,180):
        grados = ((1.0/18.0)*i)+2.5
        pulso.ChangeDutyCycle(grados)
        print("horario")
    sleep(2)		
    for i in range(180,0,-1):
        grados = ((1.0/18.0)*i)+2.5
        pulso.ChangeDutyCycle(grados)
        print("antihorario")
    sleep(2)        
    pulso.stop() 
    estadoServo = 0   

#create object from ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id = recognizer.predict(gray[y:y+h,x:x+w])
        print(Id[0])
        # Check the ID if exist 
        if(Id[0] == 1):
            Id = "Amparo"                                   
            if(estadoServo == 0):
                estadoServo = 1 - estadoServo                               
                executor.submit(moveServo)
                
        elif(Id[0] == 2):
            Id = "Bety"
            if(estadoServo == 0):
                estadoServo = 1 - estadoServo                               
                executor.submit(moveServo)
                                                    
        #If not exist, then it is Unknown
        else:            
            Id = "Desconocido"            
        
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    
# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()

# close GPIo
GPIO.cleanup()
