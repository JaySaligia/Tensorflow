#coding=utf-8
import os
import tensorflow as tf
from PIL import Image
import random

orig_picture = r"C:\零售商品分类\1.17"
gen_picture = r"C:\tmpimage"

classes = [
 '奥利奥'
,'百奇饼干'
,'百岁山矿泉水'
,'茶π饮料'
,'袋装老坛酸菜牛肉面'
,'罐装百事可乐'
,'罐装可口可乐'
,'光明酸奶'
,'红烧牛肉面'
,'可口可乐零度小'
,'昆仑山矿泉水'
,'老法饼'
,'老酸奶'
,'乐事薯片黄瓜味'
,'农夫山泉'
,'苹果醋'
,'瓶装雀巢咖啡'
,'恰恰瓜子'
,'全麦面包'
,'肉松蛋糕'
,'肉松小贝'
,'水趣多'
,'汤达人桶装面'
,'王老吉'
,'星巴克'
,'伊利纯牛奶'
,'伊利酸奶'
,'益达口香糖']

num_samples = 2913

def create_record():  
    #writer = tf.python_io.TFRecordWriter("commodity_train.tfrecords")  
    writer = tf.python_io.TFRecordWriter("commodity_test.tfrecords") 
    for index in range(len(classes)):
        count = 0;
        name = classes[index]
        print (index, name)
        class_path = orig_picture + "\\" + name + "\\"
        for img_name in os.listdir(class_path): 
          count += 1
          if count % 10 == 0:
            img_path = class_path + img_name  
            img = Image.open(img_path)  
            img = img.resize((256, 256))    #设置需要转换的图片大小
            #对图片随机裁剪为224*224
            img = img.crop((random.randint(0,25), random.randint(0,25), 224, 224))
            img_raw = img.tobytes()      #将图片转化为原生bytes    
            example = tf.train.Example(  
               features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
               }))  
            writer.write(example.SerializeToString())  
    writer.close()  

create_record()





