import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten_numbers.model')


img = cv2.imread("test.png")[:,:,0]
img = np.invert(np.array([img]))
prediction = model.predict(img)
print(f"This digit is probably a {np.argmax(prediction)}")
plt.imshow(img[0],cmap=plt.cm.binary)
plt.show()
