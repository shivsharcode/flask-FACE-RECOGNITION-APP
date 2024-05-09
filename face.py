import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

def detect_faces(frame):
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        
        top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (left+20, top-20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2 )
        
    return frame
    
    
    
