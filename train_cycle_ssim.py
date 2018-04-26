import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image 
import matplotlib.pyplot as plt 

"""
Import CycleGAN with SSIM loss class definition.
"""
from cycleGAN_ssim import CycleGAN


"""
Function to load image and rescale it to [-1,1].
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
Function to save generated image to image pools. 
"""
def save_to_pool(poolA,poolB,gen_A,gen_B,pool_size,num_im):
        
        if num_im < pool_size:
            poolA[num_im] = gen_A
            poolB[num_im] = gen_B
        
        else:
            p = random.random()
            if p > 0.5:
                indA = random.randint(0,pool_size-1)
                poolA[indA] = gen_A
            p = random.random()
            if p > 0.5: 
                indB = random.randint(0,pool_size-1)
                poolB[indB] = gen_B
        
        num_im = num_im + 1
        return poolA,poolB,num_im

"""
Function to train the network
"""
def train(cgan_net,max_img,batch_size,trainA,trainB,lr_rate,shape,pool_size):
    saver = tf.train.Saver(max_to_keep=None)
    lenA = len(trainA)
    lenB = len(trainB)
    epoch = 170
    countA = 0 
    countB = 0
    num_imgs = 0
    poolA = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))
    poolB = np.zeros((pool_size,1,shape[0],shape[1],shape[2]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while epoch < 201:
            if epoch >= 100:
                lr_rate = 0.0002 - ((epoch-100)*0.0002)/100
            
            for step in range(max_img):
                
                if countA >= lenA:
                    countA = 0
                    random.shuffle(trainA)
                
                if countB >= lenB:
                    countB = 0
                    random.shuffle(trainB)
                
                  
                imgA = get_image_new(trainA[countA],shape[0],shape[1])
                countA = countA + 1
                imgB = get_image_new(trainB[countB],shape[0],shape[1])
                countB = countB + 1
                
                imgA = np.reshape(imgA,(1,shape[0],shape[1],shape[2]))
                imgB = np.reshape(imgB,(1,shape[0],shape[1],shape[2]))
               
                
                _,genB,genA_loss,_,genA,genB_loss,cyclicA,cyclicB = sess.run([cgan_net.genA_opt,cgan_net.gen_B,cgan_net.gen_loss_A,cgan_net.genB_opt,cgan_net.gen_A,cgan_net.gen_loss_B,cgan_net.cyclicA,cgan_net.cyclicB],
                                            feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB,cgan_net.lr_rate:lr_rate})
                
                poolA,poolB,num_imgs = save_to_pool(poolA,poolB,genA,genB,pool_size,num_imgs)
                
                
                indA = random.randint(0,(min(pool_size,num_imgs)-1))
                indB = random.randint(0,(min(pool_size,num_imgs)-1))
                fakeA_img = poolA[indA]
                fakeB_img = poolB[indB]
                
                
                _,discA_loss,_,discB_loss = sess.run([cgan_net.discA_opt,cgan_net.disc_loss_A,cgan_net.discB_opt,cgan_net.disc_loss_B],
                         feed_dict={cgan_net.input_A:imgA,cgan_net.input_B:imgB,cgan_net.lr_rate:lr_rate,cgan_net.fake_pool_Aimg:fakeA_img,cgan_net.fake_pool_Bimg:fakeB_img})
                
                
                
                if step % 50 == 0:
                    print ("epoch = %r step = %r discA_loss = %r genA_loss = %r discB_loss = %r genB_loss = %r" 
                           %(epoch,step,discA_loss,genA_loss,discB_loss,genB_loss))
                    
                if step % 150 == 0:
                    images = [genA,cyclicB,genB,cyclicA]
                    for img in images:
                        img = np.reshape(img,(shape[0],shape[1],shape[2]))
                        if np.array_equal(img.max(),img.min()) == False:
                            img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
                        else:
                            img = ((img - img.min())*255).astype(np.uint8)
                        plt.imshow(img)
                        plt.show()
                print ("calc =%r" %(step))
                
            if epoch % 10 == 0:
                #change the second argument of save function to the path where you want to save model weights 
                saver.save(sess,model_dir+"try_"+str(epoch)+"\\",write_meta_graph=True)
                print ("### Model weights Saved epoch = %r ###" %(epoch))
            
            epoch = epoch + 1
        
        
          

#change model_dir to the parent directory where weights would be saved                     
model_dir = "cycleGAN\\model\\"

#change path values for dataset
trainA = glob("cycleGAN\\dataset\\monet2photo\\trainA\\*.jpg")
trainB = glob("cycleGAN\\dataset\\monet2photo\\trainB\\*.jpg")

input_shape = 128,128,3
batch_size = 1
pool_size = 50 
lr_rate = 0.0002
beta1 = 0.5
max_img = 250
tf.reset_default_graph()

cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1)

train(cgan_net,max_img,batch_size,trainA,trainB,lr_rate,input_shape,pool_size)

