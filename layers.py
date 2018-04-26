import tensorflow as tf 

"""
Helper file. To define functions related to layers of cycleGAN. 
"""


def instance_norm(inp):
    
    with tf.variable_scope("instance_norm"):
        eps = 1e-5
        mean,var = tf.nn.moments(inp,[1,2],keep_dims=True)
        scale = tf.get_variable('scale',[inp.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.02))
        offset = tf.get_variable('offset',[inp.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(inp-mean,tf.sqrt(var+eps)) + offset
        
        return out


"""
Function to define general convolution operation
"""
def general_conv(input_conv,filters=64,kernel=7,stride=1,padding='VALID',name="conv",norm = True,stddev=0.02,relu = True, alpha = 0):
    w_init = tf.truncated_normal_initializer(mean=0.0,stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    with tf.variable_scope(name):
        
        conv = tf.layers.conv2d(input_conv,filters,kernel,stride,padding,kernel_initializer=w_init,bias_initializer=b_init)
        if norm == True:
            conv = instance_norm(conv)
        
        if relu == True:
            if alpha == 0:
                conv = tf.nn.relu(conv)
            else:
                conv = tf.nn.leaky_relu(conv,alpha=alpha)
        
        return conv
    
"""
Function to define the resnet block
"""
def resnet_block(input_res,filters,name="resnet"):
    
    with tf.variable_scope(name):
        output_res = tf.pad(input_res,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
        output_res = general_conv(output_res,filters=filters,kernel=3,stride=1,padding="VALID",name="c1",norm=True,stddev=0.02,relu=True)
        output_res = tf.pad(output_res,[[0,0],[1,1],[1,1],[0,0]],mode="REFLECT")
        output_res = general_conv(output_res,filters=filters,kernel=3,stride=1,padding="VALID",name="c2",norm=True,stddev=0.02,relu=False)
        
        return tf.nn.relu(output_res + input_res)    

"""
Function to define the general de-convolution operation.
"""
def general_deconv(input_deconv,filters=64,kernel=7,stride=1,padding="VALID",name="deconv",norm=True,stddev=0.02,relu=True,alpha=0):
    w_init = tf.truncated_normal_initializer(mean=0.0,stddev=0.02)
    b_init = tf.constant_initializer(0.0)
    with tf.variable_scope(name):
        conv = tf.layers.conv2d_transpose(input_deconv,filters,kernel,stride,padding,kernel_initializer=w_init,bias_initializer=b_init)
        
        if norm == True:
            conv = instance_norm(conv)
        
        if relu == True:
            if alpha == 0:
                conv = tf.nn.relu(conv)
            else:
                conv = tf.nn.leaky_relu(conv,alpha=alpha)
        return conv



"""
Generator Architecture with 6 ResNet blocks as described in CycleGAN Paper.
"""
def generator(input_gen,name='generator'):
    
    with tf.variable_scope(name):
        
        pad_input =tf.pad(input_gen,[[0,0],[3,3],[3,3],[0,0]],mode="REFLECT")
        
        c7s1_32 = general_conv(pad_input,filters=32,kernel=7,stride=1,padding='VALID',name="conv1",norm=True,stddev=0.02,relu=True)
        d64 = general_conv(c7s1_32,filters=64,kernel=3,stride=2,padding="SAME",name="conv2",norm=True,stddev=0.02,relu=True)
        d128 = general_conv(d64,filters=128,kernel=3,stride=2,padding="SAME",name="c3",norm=True,stddev=0.02,relu=True)
        
        R128_1 = resnet_block(d128,filters=128,name="r1")
        R128_2 = resnet_block(R128_1,filters=128,name="r2")
        R128_3 = resnet_block(R128_2,filters=128,name="r3")
        R128_4 = resnet_block(R128_3,filters=128,name="r4")
        R128_5 = resnet_block(R128_4,filters=128,name="r5")
        R128_6 = resnet_block(R128_5,filters=128,name="r6")
        
        u64 = general_deconv(R128_6,filters=64,kernel=3,stride=2,padding="SAME",name="dc1",norm=True,stddev=0.02,relu=True)
        u32 = general_deconv(u64,filters=32,kernel=3,stride=2,padding="SAME",name="dc2",norm=True,stddev=0.02,relu=True)
        
        u32_pad = tf.pad(u32,[[0,0],[3,3],[3,3],[0,0]],mode="REFLECT")
        
        c7s1_3 = general_conv(u32_pad,filters=3,kernel=7,stride=1,padding="VALID",name="c4",norm=True,stddev=0.02,relu=False)
        
        output_gen = tf.nn.tanh(c7s1_3,"out_gen")
        
        return output_gen
    

"""
Discriminator Architecture as described in CycleGAN paper.
"""
def discriminator(input_disc,name="discriminator"):
    
    with tf.variable_scope(name):
        
        C64 = general_conv(input_disc,filters=64,kernel=4,stride=2,padding="SAME",name="c1",norm=False,relu=True,alpha=0.2)
        C128 = general_conv(C64,filters=128,kernel=4,stride=2,padding="SAME",name="c2",norm=True,stddev=0.02,relu=True,alpha=0.2)
        C256 = general_conv(C128,filters=256,kernel=4,stride=2,padding="SAME",name="c3",norm=True,stddev=0.02,relu=True,alpha=0.2)
        C512 = general_conv(C256,filters=512,kernel=4,stride=2,padding="SAME",name="c4",norm=True,stddev=0.02,relu=True,alpha=0.2)

        logits = general_conv(C512,filters=1,kernel=4,stride=1,padding="SAME",name="disc_logits",norm=False,relu=False)

        return logits        
        
        
        

                
        
        
