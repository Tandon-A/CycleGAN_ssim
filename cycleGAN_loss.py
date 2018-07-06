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
    
    def __init__(self,batch_size,input_shape,pool_size,beta1,loss_type):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.loss_type = loss_type
        self.lr_rate = tf.placeholder(dtype=tf.float32,shape=[],name="lr_rate")
        self.g_window = self.gaussian_window(self.input_shape[0],self.input_shape[2],0.5) #uses sigma value of 0.5 to calculate gaussian window. Used to combine l1/l2 loss with ssim loss.
        self.input_A,self.input_B, self.fake_pool_Aimg, self.fake_pool_Bimg = self.model_inputs(self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])
        
        self.gen_A,self.gen_B,self.cyclicA,self.cyclicB,self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B = self.model_arc(self.input_A,self.input_B,self.fake_pool_Aimg,self.fake_pool_Bimg)
        
        
        self.gen_loss_A, self.disc_loss_A, self.gen_loss_B,self.disc_loss_B = self.model_loss(self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B,self.input_A,self.cyclicA,self.input_B,self.cyclicB,self.loss_type,self.g_window,(self.input_shape[2]*self.batch_size))
        
        
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
        
        with tf.variable_scope("CycleGAN_loss") as scope:
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
    def model_loss(self,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B,input_A,cyclicA,input_B,cyclicB,loss_type,g_window,norm_const):
        cyclic_loss = 0
        #l2 loss
        if loss_type == "l2":
          cyc_A = tf.reduce_sum(tf.squared_difference(input_A,cyclicA))
          cyc_B = tf.reduce_sum(tf.squared_difference(input_B,cyclicB))
          cyclic_loss = cyc_A + cyc_B
        
        #ssim loss.   
        elif loss_type == "ssim":
          cyc_A = 1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]
          cyc_B = 1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]
          cyclic_loss = cyc_A + cyc_B
        
        #combination of ssim and l1 loss
        elif loss_type == "ssim_l1":
          cyc_A = 0.84*(1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.abs(input_A - cyclicA)*g_window)/norm_const)
          cyc_B = 0.84*(1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.abs(input_B - cyclicB)*g_window)/norm_const)
          cyclic_loss = cyc_A + cyc_B
        
        #combination of ssim and l2 loss (type a)
        elif loss_type == "ssim_l2_a":
          cyc_A = (1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]) + 0.00005*tf.reduce_sum(tf.squared_difference(input_A,cyclicA))
          cyc_B = (1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]) + 0.00005*tf.reduce_sum(tf.squared_difference(input_B,cyclicB))
          cyclic_loss = cyc_A + cyc_B
       
        #combination of ssim and l2 loss (type b)
        elif loss_type == "ssim_l2_b":
          cyc_A = 0.84*(1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.squared_difference(input_A,cyclicA)*g_window)/norm_const/2)
          cyc_B = 0.84*(1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.squared_difference(input_B,cyclicB)*g_window)/norm_const/2)
          cyclic_loss = cyc_A + cyc_B
        
        #combination of l1 and l2 loss
        elif loss_type == "l1_l2":
          cyc_A = tf.reduce_sum(tf.abs(input_A - cyclicA)) + tf.reduce_sum(tf.squared_difference(input_A,cyclicA))
          cyc_B = tf.reduce_sum(tf.abs(input_B - cyclicB)) + tf.reduce_sum(tf.squared_difference(input_B,cyclicB))
          cyclic_loss = cyc_A + cyc_B
          
        #combination of ssim, l1 and l2 loss (type a)
        elif loss_type == "ssim_l1l2_a":
          cyc_A =  0.84*(1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.abs(input_A - cyclicA)*g_window)/norm_const) + 0.00005*tf.reduce_sum(tf.squared_difference(input_A,cyclicA))
          cyc_B =  0.84*(1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]) + (1-0.84)*(tf.reduce_sum(tf.abs(input_B - cyclicB)*g_window)/norm_const) + 0.00005*tf.reduce_sum(tf.squared_difference(input_B,cyclicB))
          cyclic_loss = cyc_A + cyc_B
        
        #combination of ssim, l1 and l2 loss (type b)
        elif loss_type == "ssim_l1l2_b":
          cyc_A =  0.84*(1 - tf.image.ssim(input_A,cyclicA,max_val=1.0)[0]) + (2/3)*(1-0.84)*(tf.reduce_sum(tf.abs(input_A - cyclicA)*g_window)/norm_const) + (1/3)*(1 - 0.84)*(tf.reduce_sum(tf.squared_difference(input_A,cyclicA)*g_window)/norm_const/2)
          cyc_B =  0.84*(1 - tf.image.ssim(input_B,cyclicB,max_val=1.0)[0]) + (2/3)*(1-0.84)*(tf.reduce_sum(tf.abs(input_B - cyclicB)*g_window)/norm_const) + (1/3)*(1 - 0.84)*(tf.reduce_sum(tf.squared_difference(input_B,cyclicB)*g_window)/norm_const/2)
          cyclic_loss = cyc_A + cyc_B
        
        #l1 loss.
        else:
          cyclic_loss = tf.reduce_mean(tf.abs(input_A-cyclicA)) + tf.reduce_mean(tf.abs(input_B - cyclicB))
          
        disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_disc_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_disc_B,1))
        
        gen_loss_A = cyclic_loss*10 + disc_loss_B
        gen_loss_B = cyclic_loss*10 + disc_loss_A
        
        d_loss_A = (tf.reduce_mean(tf.square(fake_pool_disc_A)) + tf.reduce_mean(tf.squared_difference(real_disc_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(fake_pool_disc_B)) + tf.reduce_mean(tf.squared_difference(real_disc_B,1)))/2.0

        return gen_loss_A,d_loss_A,gen_loss_B,d_loss_B
    
    """
    Function to optimize the network. Uses Adam Optimizer.
    """
    def model_opti(self,gen_loss_A,disc_loss_A,gen_loss_B,disc_loss_B,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()
        discA_vars = [var for var in train_vars if var.name.startswith('CycleGAN_loss/discriminator_A')]
        discB_vars = [var for var in train_vars if var.name.startswith('CycleGAN_loss/discriminator_B')]
        genA_vars = [var for var in train_vars if var.name.startswith('CycleGAN_loss/generator_A')]        
        genB_vars = [var for var in train_vars if var.name.startswith('CycleGAN_loss/generator_B')]        
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            discA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_A,var_list = discA_vars)
            discB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_B,var_list = discB_vars)
            genA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_A,var_list = genA_vars)
            genB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_B,var_list = genB_vars)
    
        return discA_train_opt, discB_train_opt, genA_train_opt, genB_train_opt
    
    """
    Function to model the Gaussian Function
    """
    def gaussian_window(self,size,channels,sigma):
      gaussian = np.arange(-(size/2), size/2)
      gaussian = np.exp(-1.*gaussian**2/(2*sigma**2))
      gaussian = np.outer(gaussian, gaussian.reshape((size, 1)))	# extend to 2D
      gaussian = gaussian/np.sum(gaussian)								# normailization
      gaussian = np.reshape(gaussian, (1,size,size,1)) 	# reshape to 4D
      gaussian = np.tile(gaussian, (1,1, 1,channels))
      return gaussian
        
