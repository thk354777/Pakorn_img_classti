import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model(r"ImageClassification-main\model_computer_equipment\computer_equipment.h5")
image_path = r"ImageClassification-main\backlit-keyboard-mac.jpg"



img = image.load_img(image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0


predictions = model.predict(img_array)

# Get the class label with the highest probability
predicted_class = np.argmax(predictions[0])


class_labels = ['Keyboard_computer', 'Monitor_computer', 'Mouse_computer'] 
predicted_class_name = class_labels[predicted_class]

print("Predicted class:", predicted_class_name)

