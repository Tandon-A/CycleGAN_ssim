import tensorflow as tf
import numpy as np 

"""
Import helper functions from layers.py
"""
from layers import *


"""
Class Definition of CycleGAN with SSIM loss. 
"""
class CycleGAN:
    
    def __init__(self,batch_size,input_shape,pool_size,beta1):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.lr_rate = tf.placeholder(dtype=tf.float32,shape=[],name="lr_rate")
        self.input_A,self.input_B, self.fake_pool_Aimg, self.fake_pool_Bimg = self.model_inputs(self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])
        
        self.gen_A,self.gen_B,self.cyclicA,self.cyclicB,self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B = self.model_arc(self.input_A,self.input_B,self.fake_pool_Aimg,self.fake_pool_Bimg)
        
        
        self.gen_loss_A, self.disc_loss_A, self.gen_loss_B,self.disc_loss_B = self.model_loss(self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B,self.input_A,self.cyclicA,self.input_B,self.cyclicB)
        
        
        self.discA_opt,self.discB_opt,self.genA_opt,self.genB_opt = self.model_opti(self.gen_loss_A,self.disc_loss_A,self.gen_loss_B,self.disc_loss_B,self.lr_rate,beta1)
        
        
    """
    Function to model inputs of the network
    """
    def model_inputs(self,batch_size,height,width,channels):
        input_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="input_A")
        input_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="input_B")         
        gen_pool_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="fake_pool_Aimg") 
        gen_pool_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="fake_pool_Bimg") 
        
        return input_A,input_B,gen_pool_A,gen_pool_B
    
    
    
    """
    Function to model architecture of CycleGAN. 
    """
    def model_arc(self,input_A,input_B,fake_pool_A,fake_pool_B):
        
        with tf.variable_scope("CycleGAN_ssim") as scope:
            gen_B = generator(input_A,name="generator_A")
            gen_A = generator(input_B,name="generator_B")
            real_disc_A = discriminator(input_A,name="discriminator_A")
            real_disc_B = discriminator(input_B,name="discriminator_B")
            
            scope.reuse_variables()
            
            fake_disc_A = discriminator(gen_A,name="discriminator_A")
            fake_disc_B = discriminator(gen_B,name="discriminator_B")
            
            cyclicA = generator(gen_B,name="generator_B")
            cyclicB = generator(gen_A,name="generator_A")
            
            scope.reuse_variables()
            
            fake_pool_disc_A = discriminator(fake_pool_A,name="discriminator_A")
            fake_pool_disc_B = discriminator(fake_pool_B,name="discriminator_B")
            
            return gen_A,gen_B,cyclicA,cyclicB,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B
        
        
    """
    Function to calculate loss.
    """          
    def model_loss(self,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B,input_A,cyclicA,input_B,cyclicB):
        
        #orignal loss as defined in paper by Zhu et. al. 
        #cyclic_loss = tf.reduce_mean(tf.abs(input_A-cyclicA)) + tf.reduce_mean(tf.abs(input_B - cyclicB))
        
        #using SSIM loss to compute cyclic loss = 1 - SSIM_loss(Input Image,Generated Image) + 0.00005*L2_loss(Input Image,Generated Image)
        cyc_A = 1 - self._SSIM(input_A,cyclicA) + 0.00005*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(input_A,cyclicA),axis=3),axis=2),axis=1))
        cyc_B = 1 - self._SSIM(input_B,cyclicB) + 0.00005*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(input_B,cyclicB),axis=3),axis=2),axis=1))
        cyclic_loss = cyc_A + cyc_B
        disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_disc_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_disc_B,1))
        
        gen_loss_A = cyclic_loss*10 + disc_loss_B
        gen_loss_B = cyclic_loss*10 + disc_loss_A
        
        d_loss_A = (tf.reduce_mean(tf.square(fake_pool_disc_A)) + tf.reduce_mean(tf.squared_difference(real_disc_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(fake_pool_disc_B)) + tf.reduce_mean(tf.squared_difference(real_disc_B,1)))/2.0

        return gen_loss_A,d_loss_A,gen_loss_B,d_loss_B
    
    
    """
    Function to optimize the network. Used Adam Optimizer.
    """
    def model_opti(self,gen_loss_A,disc_loss_A,gen_loss_B,disc_loss_B,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()
        discA_vars = [var for var in train_vars if var.name.startswith('CycleGAN_ssim/discriminator_A')]
        discB_vars = [var for var in train_vars if var.name.startswith('CycleGAN_ssim/discriminator_B')]
        genA_vars = [var for var in train_vars if var.name.startswith('CycleGAN_ssim/generator_A')]        
        genB_vars = [var for var in train_vars if var.name.startswith('CycleGAN_ssim/generator_B')]        
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            discA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_A,var_list = discA_vars)
            discB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_B,var_list = discB_vars)
            genA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_A,var_list = genA_vars)
            genB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_B,var_list = genB_vars)
    
        return discA_train_opt, discB_train_opt, genA_train_opt, genB_train_opt
        
        
    """
    Function to model the Gaussian Function
    """
    def _FSpecialGauss(self,size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1
        x, y = np.mgrid[offset + start:stop, offset + start:stop]
        g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
        return g / g.sum()
    
    
    """
    Function to calculate SSIM loss per channel.
    """
    def _SSIM(self,image1, image2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
        
        _, height, width, _ = image1.shape
        size = 11
        sigma = size * filter_sigma / filter_size
        ssim = 0
        window = np.reshape(self._FSpecialGauss(size, sigma), (size, size, 1,1))
        for i in range(3):
            img1 = tf.reshape(image1[:,:,:,i],(1,128,128,1))
            img2 = tf.reshape(image2[:,:,:,i],(1,128,128,1))
            mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
            mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
            sigma11 = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID')
            sigma22 = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID')
            sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID')
            mu11 = mu1 * mu1
            mu22 = mu2 * mu2
            mu12 = mu1 * mu2
            sigma11 -= mu11
            sigma22 -= mu22
            sigma12 -= mu12
            
            # Calculate intermediate values used by both ssim and cs_map.
            c1 = (k1 * max_val) ** 2
            c2 = (k2 * max_val) ** 2
            v1 = 2.0 * sigma12 + c2
            v2 = sigma11 + sigma22 + c2
            ssim = ssim + tf.reduce_mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
        ssim = ssim / 3
        return ssim
    
    
    
        
        
        
        
