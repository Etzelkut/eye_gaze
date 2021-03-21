import os

from together import Connect_module
import cv2
import PIL.Image
import numpy as np

path_cascade = 'haarcascade_frontalface_default.xml'
path_dots = "/content/drive/MyDrive/eye_w/first_try.ckpt"
path_eye_model = "irislandmarks.pth"

projj = Connect_module(path_cascade, path_dots, path_eye_model)


from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
    image1 = cv2.imread(UPLOAD_FOLDER + filename)
    image1 = projj.forward(image1)
    im = PIL.Image.fromarray(image1)
    im.save(UPLOAD_FOLDER + filename)
    return redirect(url_for('static', filename= 'uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()