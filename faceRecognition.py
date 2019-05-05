import cv2
import sys

# Get user supplied values
# imagePath = sys.argv[1]
angry = cv2.imread("angry.png", -1)
alpha_angry = angry[:, :, 3] / 255.0

alpha_frame = 1.0 - alpha_angry


faceCascade = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
while(1):
  i=0
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
  for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      # cv2.imshow("Face " + str(i), frame[y:y+h, x:x+w])
      resized_emoition = cv2.resize(angry, (h, w)) 
      alpha_rezied_emotion = resized_emoition[:, :, 3] / 255.0
      alpha_frame = 1.0 - alpha_rezied_emotion


      # frame[y:y+h,  x:x+w] = resized_emoition

      for c in range(0, 3):
        frame[y:y+h,  x:x+w, c] = (alpha_rezied_emotion * resized_emoition[:, :, c] + alpha_frame * frame[y:y+h,  x:x+w, c])


      i= i+1    
      cv2.imshow("Emoji Frame", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()





