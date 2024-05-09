

from flask import Flask, render_template, redirect, Request, Response, url_for

import cv2 
from face import detect_faces

from simple_facerec import SimpleFacerec

app = Flask(__name__)
camera = cv2.VideoCapture(0)


sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

def generate_frames():
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces and draw rectangles and names
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                cv2.putText(frame, name, (left + 20, top - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)