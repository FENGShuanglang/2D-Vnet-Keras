# -*- coding: utf-8 -*-
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras.backend as K
#from keras import optimizers
#from keras.utils import plot_model#使用plot_mode时打开
from keras.models import Model
from keras.layers import Conv2D,PReLU,Conv2DTranspose,add,concatenate,Input,Dropout
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def dice_coef(y_true, y_pred, smooth, thresh):
    #y_pred =K.cast((K.greater(y_pred,thresh)), dtype='float32')#转换为float型
    #y_pred = y_pred[y_pred > thresh]=1.0
    y_true_f =y_true# K.flatten(y_true)
    y_pred_f =y_pred# K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f,axis=(0,1,2))
    denom =K.sum(K.square(y_true_f),axis=(0,1,2)) + K.sum(K.square(y_pred_f),axis=(0,1,2))
    return K.mean((2. * intersection + smooth) /(denom + smooth))

def dice_loss(smooth, thresh):
    def dice(y_true, y_pred):
        
        return 1-dice_coef(y_true, y_pred, smooth, thresh)
    return dice
  
  
    

def resBlock(conv,stage,keep_prob,stage_num=5):#收缩路径
    
    inputs=conv
    
    for _ in range(3 if stage>3 else stage):
        conv=PReLU()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv))
        #print('conv_down_stage_%d:' %stage,conv.get_shape().as_list())#输出收缩路径中每个stage内的卷积
    conv_add=PReLU()(add([inputs,conv]))
    #print('conv_add:',conv_add.get_shape().as_list())
    conv_drop=Dropout(keep_prob)(conv_add)
    
    if stage<stage_num:
        conv_downsample=PReLU()(Conv2D(16*(2**stage), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop))
        return conv_downsample,conv_add#返回每个stage下采样后的结果,以及在相加之前的结果
    else:
        return conv_add,conv_add#返回相加之后的结果，为了和上面输出保持一致，所以重复输出
        
def up_resBlock(forward_conv,input_conv,stage):#扩展路径
    
    conv=concatenate([forward_conv,input_conv],axis = -1)
    print('conv_concatenate:',conv.get_shape().as_list())
    for _ in range(3 if stage>3 else stage):
        conv=PReLU()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv))
        print('conv_up_stage_%d:' %stage,conv.get_shape().as_list())#输出扩展路径中每个stage内的卷积
    conv_add=PReLU()(add([input_conv,conv]))
    if stage>1:
        conv_upsample=PReLU()(Conv2DTranspose(16*(2**(stage-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(conv_add))
        return conv_upsample
    else:
        return conv_add

def vnet(pretrained_weights = None,input_size = (256,256,1),num_class=1,is_training=True,stage_num=5,thresh=0.5):#二分类时num_classes设置成1，不是2，stage_num可自行改变，也即可自行改变网络深度
    keep_prob = 1.0 if is_training else 1.0#不使用dropout
    features=[]
    input_model = Input(input_size)
    x=PReLU()(Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input_model))
    
    for s in range(1,stage_num+1):
        x,feature=resBlock(x,s,keep_prob,stage_num)#调用收缩路径
        features.append(feature)
        
    conv_up=PReLU()(Conv2DTranspose(16*(2**(s-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(x)) 
    
    for d in range(stage_num-1,0,-1):
        conv_up=up_resBlock(features[d-1],conv_up,d)#调用扩展路径
    if num_class>1:
        conv_out=Conv2D(num_class, 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
    else:
        conv_out=Conv2D(num_class, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
    
    
    
    
    model=Model(inputs=input_model,outputs=conv_out)
    print(model.output_shape)
    
    model_dice=dice_loss(smooth=1e-5,thresh=0.5)
    model.compile(optimizer = Nadam(lr = 2e-3), loss = model_dice, metrics = ['accuracy'])
    
    #plot_model(model, to_file='model.png')
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model
#model=vnet(input_size = (512,1024,1),num_classes=1,is_training=True,stage_num=5)



