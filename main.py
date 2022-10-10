from tensorflow.python import keras
from iocluster import *
model_path = 'disp.h5'
image_path = '1010-5066.txt.png' #The test file is in the departion.zip compressed file
model = keras.models.load_model(model_path)
a1 = predict_task(image_path,model)
print('the predict number is {}'.format(a1))
