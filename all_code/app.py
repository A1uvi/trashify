from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.preprocessing import image
import numpy as np

from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.preprocessing import image
import numpy as np
import cv2

from loadingModel import process, takeimg

model = load_model('Trashify/Garbage.h5')
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

probclassC1, predclassC1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testCardboard.jpg")
probclassC2, predclassC2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testCardboard2.jpg")
probclassG1, predclassG1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testGlass.jpg")
probclassG2, predclassG2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testGlass2.jpg")
probclassM1, predclassM1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testMetal.jpg")
probclassM2, predclassM2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testMetal2.jpg")
probclassPR1, predclassPR1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testPaper.jpg")
probclassPR2, predclassPR2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testPaper2.jpg")
probclassPL1, predclassPL1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testPlastic.jpg")
probclassPL2, predclassPL2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testPlastic2.jpg")
probclassT1, predclassT1 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testTrash.jpg")
probclassT2, predclassT2 = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/portfolio/testTrash2.jpg")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

@app.route("/webcam")
def webcam():
    return render_template("predictions.html", vals = False)

@app.route("/predictions")
def predictions():
    takeimg("C:/Users/alvia/My_Projects/Trashify/static/img/userinput/captured.jpg")
    probclass, predclass = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/userinput/captured.jpg")
    return render_template("predictions.html", vals = True, pred = predclass, prob = probclass, img_src="C:/Users/alvia/My_Projects/Trashify/static/img/userinput/captured.jpg")

@app.route("/predictionsportfolio")
def predictionsportfolio():
    return render_template("predsportfolio.html",
                            testCardboardClass = predclassC1, testCardboardProb = probclassC1,
                            testCardboardClassII = predclassC2, testCardboardProbII = probclassC2,
                            testGlassClass = predclassG1, testGlassProb = probclassG1,
                            testGlassClassII = predclassG2, testGlassProbII = probclassG2,
                            testMetalClass = predclassM1, testMetalProb = probclassM1,
                            testMetalClassII = predclassM2, testMetalProbII = probclassM2,
                            testPaperClass = predclassPR1, testPaperProb = probclassPR1,
                            testPaperClassII = predclassPR2, testPaperProbII = probclassPR2,
                            testPlasticClass = predclassPL1, testPlasticProb = probclassPL1,
                            testPlasticClassII = predclassPL2, testPlasticProbII = probclassPL2,
                            testTrashClass = predclassT1, testTrashProb = probclassT1,
                            testTrashClassII = predclassT2, testTrashProbII = probclassT2)

if __name__ == '__main__':
    app.run()