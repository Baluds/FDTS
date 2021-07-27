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
from datetime import date, datetime

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
obde='none'
cridict={}
weapdict={}
from flask import *
from flask import Markup

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)
time.sleep(2.0)

@app.route('/')
def basic():
    return render_template("message.html")

def load_yolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs


def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids

def rfa():
    global video_capture, outputFrame, lock,crim,criname,cridict,obde
    

# Load a sample picture and learn how to recognize it.
    ray_image = face_recognition.load_image_file("images/ray.jfif")
    ray_face_encoding = face_recognition.face_encodings(ray_image)[0]

# Load a second sample picture and learn how to recognize it.
    alex_image = face_recognition.load_image_file("images/alex.jpg")
    alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

    aaron_image = face_recognition.load_image_file("images/aaron.jpg")
    aaron_face_encoding = face_recognition.face_encodings(aaron_image)[0]

    anna_image = face_recognition.load_image_file("images/anna.jpg")
    anna_face_encoding = face_recognition.face_encodings(anna_image)[0]

    benedict_image = face_recognition.load_image_file("images/benedict.jpg")
    benedict_face_encoding = face_recognition.face_encodings(benedict_image)[0]

    foxx_image = face_recognition.load_image_file("images/foxx.jpg")
    foxx_face_encoding = face_recognition.face_encodings(foxx_image)[0]

    hardy_image = face_recognition.load_image_file("images/hardy.jpg")
    hardy_face_encoding = face_recognition.face_encodings(hardy_image)[0]

    zac_image = face_recognition.load_image_file("images/zac.jpg")
    zac_face_encoding = face_recognition.face_encodings(zac_image)[0]


# Create arrays of known face encodings and their names
    known_face_encodings = [
        ray_face_encoding,
        alex_face_encoding,
        aaron_face_encoding,
        anna_face_encoding,
        benedict_face_encoding,
        foxx_face_encoding,
        hardy_face_encoding,
        zac_face_encoding
    ]
    known_face_names = [
        "Raymond",
        "Alex",
        "Aaron",
        "Anna",
        "Benedict",
        "Foxx",
        "Hardy",
        "Zac"
    
    ]
    count=0
    model, classes, colors, output_layers = load_yolo()
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
                    cv2.imwrite('static/crim_found/'+name+'.jpg', crim)
                    dateTimeObj = datetime.now()
                    dateStr = dateTimeObj.strftime("%d %b %Y ")
                    timeStr = dateTimeObj.strftime("%I:%M %p")
                    db.child(name).update({"Date": timeStr+", "+dateStr})
                    criname=name
                    if name not in cridict:
                        cridict[name]={}
                    cridict[name]['img'] = 'static/crim_found/'+name+'.jpg'
                    #crilist.append(name)
                #count=0
            #(flag, encodedImage) = cv2.imencode(".jpg", frame)
        # Display the resulting image
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                #print(label)
                with lock:
                    if label=='Gun' or label =='Rifle':
                        obde=label
                        weap=frame.copy()
                        cv2.imwrite('static/crim_found/'+obde+'.jpg', weap)
                        dateTimeObj = datetime.now()
                        dateStr = dateTimeObj.strftime("%d %b %Y ")
                        timeStr = dateTimeObj.strftime("%I:%M %p")
                        db.child(obde).update({"Date": timeStr+", "+dateStr})
                        if obde not in weapdict:
                            weapdict[obde]={}
                        weapdict[obde]['img'] = 'static/crim_found/'+obde+'.jpg'

                try:
                    color = colors[i]
                except:
                    print("error occurred")
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 5), font, 1, color, 1)
	    #img=cv2.resize(frame, (800,600))
	    #cv2.imshow("Image", frame) 
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
    global criname
    #return redirect(url_for('static',filename='crim_found/crim1.jpg'))
    imsr = "static/crim_found/"+ criname +".jpg"
    return imsr

@app.route("/sucess")
def sucess():
    global criname,cridict
    ag='-'
    sex='-'
    hei='-'
    noc='-'
    fn='-'
    da='-'
    addre='-'
    dan='-'
    if(criname!='none'):
        age=db.child(criname).get()
        age=age.val()
        cridict[criname]['address']=addre=list(age.values())[0]
        cridict[criname]['age']=ag=list(age.values())[1]
        cridict[criname]['danger']=dan=list(age.values())[2]
        cridict[criname]['date']=da=list(age.values())[3]
        cridict[criname]['hei']= hei=list(age.values())[4]
        cridict[criname]['fn']=fn=list(age.values())[5]
        cridict[criname]['noc']=noc=list(age.values())[6]
        cridict[criname]['sex']=sex=list(age.values())[7] 
    return render_template('sucess.html', ag=ag,cri=fn,se=sex,he=hei,noc=noc,da=da,addre=addre,dan=dan)

@app.route('/login', methods=["POST", "GET"])
def login():
    global criname
    criname='none'
    message = Markup('')
    if request.method == "POST":
        email = request.form["uname"]
        password = request.form["psw"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            user = auth.refresh(user['refreshToken'])
            user_id = user['idToken']
            return redirect(url_for('basic'))
        except: 
            message = Markup('<div style="color: red; margin-top: 5px;"> Incorrect Password! Try again </div>')
    return render_template("login.html", message=message)

@app.route('/logout', methods=["POST", "GET"])
def logout():
    global criname
    criname='none'
    return redirect(url_for('login'))

@app.route('/forgot', methods=["POST", "GET"])
def forgot():
    passmsg = Markup('')
    if request.method == "POST":
        email = request.form["email"]
        try:
            auth.send_password_reset_email(email)
            passmsg=Markup('<div style="color: #008140;"> Password reset link has been sent </div>')  
        except Exception as e:
            passmsg=Markup('<div style="color: red;"> We couldn’t find an account with that e-mail, Try Again </div>')
           # print(e)
    return render_template("forgot.html",passmsg=passmsg)

@app.route('/clist', methods=["POST", "GET"])
def clist():
    global cridict
    msg=""
    if not cridict:
        msg="No Fugitives recognised yet"
    return render_template("clist.html",cridict=cridict,msg=msg)

@app.route("/objectsucess")
def objectsucess():
    global obde,weapdict
    wea='-'
    cla='-'
    dan='-'
    dat='-'
    addre='-'
    if((obde!='none') and (obde!='Fire') ):
        headi=db.child(obde).get()
        headi=headi.val()
        weapdict[obde]['address']=addre=list(headi.values())[0]
        weapdict[obde]['class']=cla=list(headi.values())[1]
        weapdict[obde]['danger']=dan=list(headi.values())[2]
        weapdict[obde]['date']=dat=list(headi.values())[3]
        weapdict[obde]['weapon']= wea=list(headi.values())[4] 
    return render_template('objectsucess.html', addre=addre,cla=cla,dat=dat,wea=wea,dan=dan)

@app.route("/table1")
def table1():
    global obde
    #return redirect(url_for('static',filename='crim_found/crim1.jpg'))
    imsr = "static/crim_found/"+ obde +".jpg"
    return imsr

@app.route('/glist', methods=["POST", "GET"])
def glist():
    global weapdict
    msg=""
    if not weapdict:
        msg="No Weapons recognised yet"
    return render_template("glist.html",weapdict=weapdict,msg=msg)

if __name__ == '__main__':
    threading.Thread(target=rfa,daemon = True).start()
    app.run(debug=True,threaded=True,use_reloader=False)

video_capture.release()
