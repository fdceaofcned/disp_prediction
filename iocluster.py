import os
import shutil
import tensorflow as tf
from tensorflow.python import keras
import cv2
import numpy as np
# we test process the image via tensoflow, create image with mark
def cut_str(str_aim,key): #
    res = list(filter(None,str_aim.split(key))) #
    return res
def output_geo(output_list,save_path): # 
    output = open(save_path,'w',encoding='utf-8')
    for row in output_list:
        rowtxt = '{}'.format(row)
        output.write(rowtxt)
        output.write('\n')
    output.close()
    return print('txt file output')
def read_manual(file_path,separator,target_list): # sample target_list = [0,1,2]
    import pandas as pd
    data = pd.read_csv(file_path,header=None)
    raw_vsmod = []
    for i in range(0,len(data),1):
        raw_vsmod.append(cut_str(data.loc[i][0],separator))
    output_list = []
    for k in target_list:
        per_row = [float(data[k]) for data in raw_vsmod]
        output_list.append(per_row)
    return output_list
def divide_data(raw_data,divide_aim): #
    import numpy as np
    divide_index = np.zeros([len(raw_data)])
    for i in range(0,len(divide_aim),1):
        area_code = np.array(raw_data,np.float16)
        area_code[area_code<=divide_aim[i]] = 0
        area_code[area_code>divide_aim[i]] = 1
        divide_index = divide_index + area_code
    return divide_index
def divide_data_numba(raw_data,divide_aim): # numba support
    divide_index = np.array([0]*len(raw_data))
    num = len(divide_aim)
    for i in range(0,num,1):
        area_code = raw_data
        area_code = np.where(area_code<=divide_aim[i],0,1)
        divide_index = divide_index + area_code
    return divide_index
def two_divide_image(accept_path,source_path,save_path): # this function to divide image of disp as revise_image and accept_image
    source_list = os.listdir(source_path)
    accept_list = os.listdir(accept_path)
    revise_list = [x for x in source_list if x not in accept_list]
    for i in revise_list:
        source = source_path + i
        target = save_path + i
        shutil.copy(source,target)
        print('divide over')
    return accept_list,revise_list
def laod_image_tf(path): # load the image of disp for tensorflow
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_png(img_raw,channels=3)
    img_tensor = tf.image.resize(img_tensor,[374,1060])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # img_tensor = img_tensor/255
    return img_tensor
def predict_task(image_path,model): # seal function for predict_model,divide_threshold
    # model = keras.models.load_model(model_path)
    img_tensor = laod_image_tf(image_path)
    img_tensor = tf.expand_dims(img_tensor,axis=0)
    result = model.predict(img_tensor)[0][0]
    return result
def read_vsmod(file_path): # read the result for surfTomo software
    import pandas as pd
    data = pd.read_csv(file_path,header=None)
    raw_vsmod = []
    for i in range(0,len(data),1):
        raw_vsmod.append(cut_str(data.loc[i][0],' '))
    lon = [float(lon[0]) for lon in raw_vsmod]
    lat = [float(lat[1]) for lat in raw_vsmod]
    deepth = [float(deep[2]) for deep in raw_vsmod]
    vs = [float(vs[3]) for vs in raw_vsmod]
    return lon,lat,deepth,vs
def read_dc(file_path): # read the disp curve data
    import pandas as pd
    import numpy as np
    data = pd.read_csv(file_path,header=None)
    raw_dc = []
    for i in range(2,len(data),1):
        raw_dc.append(cut_str(data.loc[i][0],' '))
    x_period = [float(lon[0]) for lon in raw_dc]
    y_v = [float(lat[1]) for lat in raw_dc]
    return x_period,y_v
def load_curve_data(path):
    import numpy as np
    import os
    file_list = os.listdir(path)
    np_list = []
    for i in file_list:
        x_period,y_v = read_dc(path + i)
        np_list.append(y_v)
    return np.array(np_list)
def vs_pos(period_list,v_list,period_range,v_range,image_size): # 
    v_fix,p_fix = [],[]
    for k in range(0,len(v_list),1):
        if v_list[k] > 0:
            v_fix.append(v_list[k])
            p_fix.append(period_list[k])
    v_list = v_fix
    period_list = p_fix
    red_x,red_y = [],[]
    for i in range(0,len(period_list),1):
        point_xpix = round(image_size[0] * (period_list[i] - period_range[0])/(period_range[1] - period_range[0]))
        point_ypix = round(image_size[1] * (1 - (v_list[i] - v_range[0])/(v_range[1] - v_range[0])))
        red_x.append(point_xpix)
        red_y.append(point_ypix)
    return red_x,red_y
def rect_matrix(red_x,red_y,len_s): # 
    import numpy as np
    red_x = np.array(red_x)
    red_y = np.array(red_y)
    fix_x,fix_y = [],[]
    for i in range(-len_s,len_s+1,1):
        for k in range(-len_s,len_s+1,1):
            fix_x.append(i)
            fix_y.append(k)
    new_x,new_y = [],[]
    for m in range(0,len(fix_x),1):
        new_x = np.concatenate((new_x,red_x + fix_x[m]))
        new_y = np.concatenate((new_y,red_y + fix_y[m]))
    return new_x.astype(int),new_y.astype(int)