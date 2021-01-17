from flask import (
    Flask,
    render_template,
    request,
    url_for,
    redirect
    )
from model import Model
import os
from pathlib import Path

app = Flask(__name__)

serializedModelDirectory = './resnet50/'
model = Model(serializedModelDirectory)

imgPath = Path('images')

def deleteImages():
    for f in os.listdir('static'/imgPath):
        os.remove('static'/imgPath/f)

@app.route('/', methods=['GET'])
def upload():
    deleteImages()
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def result():
    deleteImages()
    uploaded_file = request.files['file']
    if not uploaded_file.filename:
        return redirect(url_for('/'))
    uploaded_file.save('static'/imgPath/uploaded_file.filename)
    output = model.predict('static'/imgPath/uploaded_file.filename)
    return render_template('index.html', img=imgPath/uploaded_file.filename, output=output)