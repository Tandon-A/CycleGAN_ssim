import tensorflow as tf 
import numpy as np 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt 

"""
Import model definition
"""
from cycleGAN_ssim import CycleGAN

"""
Function to load image from path and rescale it to [-1,1]. 
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)
    return image


"""
Function to test CycleGAN for first ten images from test set.
"""
def test(cgan_net,testA,testB):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #change second argument to path where weights have been saved 
        saver.restore(sess,model_dir+"try_200\\")
        
        for i in range(10):
            imgA = np.reshape(get_image_new(testA[i+10],128,128),(1,128,128,3))
            imgB = np.reshape(get_image_new(testB[i+10],128,128),(1,128,128,3))
        
            genA,genB,cycA,cycB = sess.run([cgan_net.gen_A,cgan_net.gen_B,cgan_net.cyclicA,cgan_net.cyclicB],
                                       feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB})
            images = [imgA,genA,cycA,imgB,genB,cycB]
            for img in images:
                img = np.reshape(img,(128,128,3))
                if np.array_equal(img.max(),img.min()) == False:
                    img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
                else:
                    img = ((img - img.min())*255).astype(np.uint8)
                        
                plt.imshow(img)
                plt.show()
            print (i)
            

                    

#change model_dir to parent directory of model weights folder                    
model_dir = "cycleGAN\\model_sw_250\\"
#change testA and testB path 
testA = glob("cycleGAN\\dataset\\monet2photo\\testA\\*.jpg")
testB = glob("cycleGAN\\dataset\\monet2photo\\testB\\*.jpg")
input_shape = 128,128,3
batch_size = 1
pool_size = 50 
lr_rate = 0.0002
beta1 = 0.5
max_img = 100
tf.reset_default_graph()

cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1)
test(cgan_net,testA,testB)
