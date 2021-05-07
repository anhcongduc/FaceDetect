from flask import Flask, request,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import FaceFuntions
import dlib
from face_detector import get_face_detector
from face_landmarks import get_landmark_model
from tensorflow.keras.models import load_model

detector = dlib.get_frontal_face_detector()
face_model = get_face_detector()
landmark_model = get_landmark_model()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
model = load_model("models/lenet_smiles.hdf5")

left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DOWNLOAD_FOLDER'] = 'downloads/'
app.config['IDCARD_FOLDER']='IDCard/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/FaceDetection', methods=['POST'])
def FaceDetection():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    for image_path in TEST_IMAGE_PATHS:
        img=cv2.imread(image_path)
        faces=FaceFuntions.FaceDetection(img,face_model)
        return jsonify({"Detect":str(len(faces)>0)},{"Face":str(faces)}),200
    return jsonify({"error_code":"Fail to Detect Face"}),400

@app.route('/SmileDetection', methods=['POST'])
def SmileDetection():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    for image_path in TEST_IMAGE_PATHS:
        img=cv2.imread(image_path)
        faces=FaceFuntions.FaceDetectionForBlinkAndSmile(img,detector)
        for face in faces:
            return jsonify({"Smile":str(FaceFuntions.SmileDetection(img[face.top():face.bottom(),face.left():face.right()],model))}),200
    return jsonify({"error_code":"Fail to Detect Smile"}),400

@app.route('/BlinkDetection', methods=['POST'])
def BlinkDetection():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    for image_path in TEST_IMAGE_PATHS:
        img=cv2.imread(image_path)
        faces=FaceFuntions.FaceDetectionForBlinkAndSmile(img,detector)
        for face in faces:
            return jsonify({"Blink":str(FaceFuntions.BlinkDetection(img,face,left_eye_landmarks,right_eye_landmarks,predictor))}),200
    return jsonify({"error_code":"Fail to Detect Blink"}),400

@app.route('/HeadPoseDetection', methods=['POST'])
def HeadPoseDetection():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    for image_path in TEST_IMAGE_PATHS:
        img=cv2.imread(image_path)
        faces=FaceFuntions.FaceDetection(img,face_model)
        for face in faces:
            return jsonify({"Head Pose":FaceFuntions.HeadPoseDetection(img,landmark_model,face)}),200
    return jsonify({"error_code":"Fail to Detect Head Pose"}),400

@app.route('/GenderDetection', methods=['POST'])
def GenderDetection():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    for image_path in TEST_IMAGE_PATHS:
        img=cv2.imread(image_path)
        faces=FaceFuntions.FaceDetectionForBlinkAndSmile(img,detector)
        for face in faces:
            return jsonify({"Gender":FaceFuntions.GenderDetection(img[face.top():face.bottom(),face.left():face.right()])}),200
    return jsonify({"error_code":"Fail to Detect Gender"}),400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
