from flask import Flask, render_template, request, redirect, url_for
import pymongo
from io import BytesIO
from PIL import Image
import ssl
import numpy
import face_recognition
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    name = request.form["content"]
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        #print(uploaded_file.filename)
        image_file = BytesIO()
        uploaded_file.save(image_file)
        image_file.seek(0)
        uploaded_file =Image.open(image_file)
        uploaded_file =numpy.array(uploaded_file)
        uploaded_file =face_recognition.face_encodings(uploaded_file)[0]
        #print(uploaded_file)
        uploaded_file =uploaded_file.tolist()
        #print(uploaded_file)
        client = pymongo.MongoClient("mongodb+srv://basil:project%40123@dab.iz1ar.mongodb.net/DAB?retryWrites=true&w=majority", ssl_cert_reqs=ssl.CERT_NONE)
        #client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["DAB"]
        identity=db["identity"]
        dict={"name": name , "image" : uploaded_file}
        identity.insert_one(dict)


    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(port=8000,debug=True)