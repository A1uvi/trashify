from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.preprocessing import image
import numpy as np
import cv2

def process(model, labels, img_path):
  img = load_img(img_path, target_size=(300, 300))
  img = img_to_array(img, dtype=np.uint8)
  img=np.array(img)/255.0
  p=model.predict(img[np.newaxis, ...])
  img_prob = np.max(p[0], axis=-1)
  predicted_class = labels[np.argmax(p[0], axis=-1)]
  return round(img_prob*100,2), predicted_class


def takeimg(path):
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
            # Save the captured frame to disk
            cv2.imwrite(path, frame)
            print("Photo saved!")
            break

    cap.release()
    cv2.destroyAllWindows()