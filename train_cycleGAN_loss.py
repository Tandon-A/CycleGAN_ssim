import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image 
import os

"""
Import CycleGAN class definition.
"""
from cycleGAN_loss import CycleGAN


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
Function to load training images.
"""
def get_data(trainA,trainB,width,height):
  tr_A = []
  tr_B = []
  for i in range(len(trainA)):
    tr_A.append(get_image_new(trainA[i],width,height))
    if i % 200 == 0:
      print ("getting trainA = %r" %(i))
  for i in range(len(trainB)):
    tr_B.append(get_image_new(trainB[i],width,height))
    if i % 200 == 0:
      print ("getting trainB = %r" %(i))
  tr_A = np.array(tr_A)
  tr_B = np.array(tr_B)
  print ("Completed loading training data. DomainA = %r , DomainB = %r" %(tr_A.shape,tr_B.shape))
  return tr_A,tr_B
    
    
"""
Function to save generated image to image pools. 
"""
def save_to_pool(poolA,poolB,gen_A,gen_B,pool_size,num_im):
        
        if num_im < pool_size:
            poolA[num_im] = gen_A
            poolB[num_im] = gen_B
            num_im = num_im + 1
        
        else:
            p = random.random()
            if p > 0.5:
                indA = random.randint(0,pool_size-1)
                poolA[indA] = gen_A
            p = random.random()
            if p > 0.5: 
                indB = random.randint(0,pool_size-1)
                poolB[indB] = gen_B
                
        return poolA,poolB,num_im

"""
Function to train the network
"""
def train(cgan_net,max_img,batch_size,trainA,trainB,lr_rate,shape,pool_size,model_dir):
    saver = tf.train.Saver(max_to_keep=None)
    lenA = len(trainA)
    lenB = len(trainB)
    epoch = 0
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
                    np.random.shuffle(trainA)
                
                if countB >= lenB:
                    countB = 0
                    np.random.shuffle(trainB)
                
                  
                imgA = trainA[countA]
                countA = countA + 1
                imgB = trainB[countB]
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
                
                
                #tlogging training loss details 
                if step % 50 == 0 and epoch % 5 == 0:
                    print ("epoch = %r step = %r discA_loss = %r genA_loss = %r discB_loss = %r genB_loss = %r" 
                           %(epoch,step,discA_loss,genA_loss,discB_loss,genB_loss))
                           
            epoch = epoch + 1
        
        saver.save(sess,model_dir,write_meta_graph=True)
        print ("### Model weights Saved epoch = %r ###" %(epoch))
        
          


def main(_): 
    if not os.path.exists(FLAGS.data_path):
        print ("Training Path doesn't exist")
    else:
            
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
    
        trainA = glob(FLAGS.data_path+"//trainA//"+FLAGS.input_fname_pattern)
        trainB = glob(FLAGS.data_path+"//trainB//"+FLAGS.input_fname_pattern)
        tr_imgA, tr_imgB = get_data(trainA,trainB,128,128)
        input_shape = 128,128,3
        batch_size = 1
        pool_size = 50 
        lr_rate = 0.0002
        beta1 = 0.5
        max_img = 100    
        # change loss type. Options - l1, l2, ssim, ssim_l1, ssim_l2_a, ssim_l2_b, ssim_l1l2_a, ssim_l1l2_b, l1_l2
        loss_type = FLAGS.loss_type
        tf.reset_default_graph()
        
        cgan_net = CycleGAN(batch_size,input_shape,pool_size,beta1,loss_type)
        
        train(cgan_net, max_img, batch_size, tr_imgA, tr_imgB, lr_rate, input_shape, pool_size, os.path.join(FLAGS.model_dir, 'model_' + loss_type))



flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Path to parent directory of trainA and trainB folder")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("model_dir","CycleGAN_model","Directory name to save checkpoints")
flags.DEFINE_string("loss_type","l1","Loss type with which cycleGAN is to be trained")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()

