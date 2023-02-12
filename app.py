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


model = load_model('C:/Users/alvia/My_Projects/garbageRecognition/Garbage.h5')
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}


def process(model, labels, img_path):
  img = load_img(img_path, target_size=(300, 300))
  img = img_to_array(img, dtype=np.uint8)
  img=np.array(img)/255.0
  p=model.predict(img[np.newaxis, ...])
  img_prob = np.max(p[0], axis=-1)
  predicted_class = labels[np.argmax(p[0], axis=-1)]
  img_prob = round(img_prob*100, 2)
  return img_prob, predicted_class

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
    return render_template("webcam.html")

@app.route("/predictions")
def predictions():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not capture frame")
            break
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("c"):
            print("Save the captured frame to disk")
            path = "C:/Users/alvia/My_Projects/Trashify/static/img/userinput/captured.jpg"
            cv2.imwrite(path, frame)
            print("Photo saved!")
            break

    cap.release()
    cv2.destroyAllWindows()

    probclass, predclass = process(model,labels,"C:/Users/alvia/My_Projects/Trashify/static/img/userinput/captured.jpg")
    return render_template("predictions.html", pred = predclass, prob = probclass)

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