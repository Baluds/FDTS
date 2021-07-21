from logging import debug
from os import name
import face_recognition
import cv2
import numpy as np
import threading
import pyrebase
import time
import argparse
from kay import config

firebase = pyrebase.initialize_app(config)

db = firebase.database()
auth = firebase.auth()

#db.child("criminals").child("crime").push({"name":"Ray"})
#db.child("Raymond").update({"Age": "49"})
#age= db.child("Raymond").get()
#va=age.val()
#print(list(va.values())[0])

outputFrame = None
lock = threading.Lock()
crim=None
criname='none'
from flask import *

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

@app.route('/')
def basic():
    return render_template("message.html")

def rfa():
    global video_capture, outputFrame, lock,crim,criname
    

# Load a sample picture and learn how to recognize it.
    ray_image = face_recognition.load_image_file("images/ray.jfif")
    ray_face_encoding = face_recognition.face_encodings(ray_image)[0]

# Load a second sample picture and learn how to recognize it.
    vin_image = face_recognition.load_image_file("images/vin.jfif")
    vin_face_encoding = face_recognition.face_encodings(vin_image)[0]

    pun_image = face_recognition.load_image_file("images/pun.jpg")
    pun_face_encoding = face_recognition.face_encodings(pun_image)[0]

# Create arrays of known face encodings and their names
    known_face_encodings = [
        ray_face_encoding,
        vin_face_encoding,
        pun_face_encoding
    ]
    known_face_names = [
        "Raymond",
        "Vin",
        "Punith"
    
    ]
    count=0
    while True:
    # Grab a single frame of video
        ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        name = ''
    # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                count=1
                name = known_face_names[best_match_index]
                #if(count==1):
                 #   from app import bcall
                  #  bcall(name)
            else:
                count=0
       # break
            #print(name)
        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            with lock:
                if(count==1):
                    crim=frame.copy()
                    cv2.imwrite('static/crim_found/crim1'+'.jpg', crim)
                    criname=name
                #count=0
            #(flag, encodedImage) = cv2.imencode(".jpg", frame)
    # Display the resulting image  
        with lock:
            outputFrame = frame.copy()

def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

#def crimgen():
 #   while True:
        

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
	# return the response generated along with the specific media
	# type (mime type)
#def bcall(uname):
#    return render_template('message.html',name=uname)  

@app.route("/table")
def table():
    #return redirect(url_for('static',filename='crim_found/crim1.jpg'))
    imsr = "static/crim_found/crim1.jpg"
    return imsr

@app.route("/sucess")
def sucess():
    global criname
    ag='-'
    sex='-'
    hei='-'
    noc='-'
    toc='-'
    fn='-'
    if(criname!='none'):
        age=db.child(criname).get()
        age=age.val()
        ag=list(age.values())[0]
        sex=list(age.values())[4]
        hei=list(age.values())[1]
        noc=list(age.values())[3]
        toc=list(age.values())[5]
        fn=list(age.values())[2]
    return render_template('sucess.html', ag=ag,cri=fn,se=sex,he=hei,noc=noc,toc=toc)

@app.route('/login', methods=["POST", "GET"])
def login():
    global criname
    criname='none'
    message = ""
    if request.method == "POST":
        email = request.form["uname"]
        password = request.form["psw"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user = auth.refresh(user['refreshToken'])
            user_id = user['idToken']
            return redirect(url_for('basic'))
        except:
            message = "Incorrect Password! Try again"
    return render_template("login.html", message=message)

@app.route('/logout', methods=["POST", "GET"])
def logout():
    global criname
    criname='none'
    return redirect(url_for('login'))

@app.route('/forgot', methods=["POST", "GET"])
def forgot():
    return render_template("forgot.html")

if __name__ == '__main__':
    threading.Thread(target=rfa,daemon = True).start()
    app.run(debug=True,threaded=True,use_reloader=False)

video_capture.release()
