from tensorflow.python import keras
from iocluster import *
model_path = 'disp.h5'
image_path = 'accept_pure_img/1010-5066.txt.png'
model = keras.models.load_model(model_path)
a1 = predict_task(image_path,model)
print('the predict number is {}'.format(a1))