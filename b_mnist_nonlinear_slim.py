from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import seaborn as sns
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder(x,b_dim,reuse=False):
    with tf.variable_scope("encoder", reuse = reuse):
        h2 = slim.stack(x, slim.fully_connected,[200,200],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, b_dim, activation=None)
    return log_alpha


def decoder(b,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder", reuse = reuse):
        h2 = slim.stack(b ,slim.fully_connected,[200,200],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, x_dim, activation=None)
    return log_alpha


def fun1(x_star,log_alpha_b,E,axis_dim=1,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*(d_x or d_b)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return elbo,  N*K
    '''
    #log q(E|x_star), b_dim is global
    #log_alpha_b = encoder(x_star,b_dim,reuse=reuse_encoder)
    log_q_b_given_x = bernoulli_loglikelihood(E, log_alpha_b)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b_given_x = tf.reduce_sum(log_q_b_given_x , axis=axis_dim)

    #log p(E)
    log_p_b = bernoulli_loglikelihood(E, tf.zeros_like(E)) 
    log_p_b = tf.reduce_sum(log_p_b, axis=axis_dim)
    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, log_alpha_x)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    # neg-ELBO
    return log_q_b_given_x - (log_p_x_given_b + log_p_b) 
    
def evidence(sess,data,elbo, batch_size = 100, S = 100, total_batch = None):
    '''
    For correct use:
    ELBO for x_i must be calculated by SINGLE z sample from q(z|x_i)
    '''
    #from scipy.special import logsumexp    
    if total_batch is None:
        total_batch = int(data.num_examples / batch_size)
        
    avg_evi = 0
    for j in range(total_batch):
        test_xs = data.next_batch(batch_size)         
        elbo_accu = np.empty([batch_size,0])
        for i in range(S):
            elbo_i = sess.run(elbo,{x:test_xs})
            elbo_accu = np.append(elbo_accu,elbo_i,axis=1)
        
        evi0 = sess.run(tf.reduce_logsumexp(elbo_accu,axis = 1))
        evi = np.mean(evi0 - np.log(S))
        avg_evi += evi / total_batch
    return avg_evi 

#%%    

tf.reset_default_graph() 

b_dim = 200; x_dim = 784
eps = 1e-10

lr=tf.constant(0.0001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#logits for bernoulli, encoder q(b|x) = log Ber(b|log_alpha_b)
log_alpha_b = encoder(x_binary,b_dim)  #N*d_b 

q_b = Bernoulli(logits=log_alpha_b) #sample K_b \bv
b_sample = tf.cast(q_b.sample(),tf.float32) #K_b*N*d_b, accompanying with encoder parameter, cannot backprop

#compute decoder p(x|b), gradient of decoder parameter can be automatically given by loss
neg_elbo = fun1(x_binary,log_alpha_b,b_sample,reuse_encoder=True,reuse_decoder= False)[:,np.newaxis]
gen_loss = tf.reduce_mean(neg_elbo) #average over N


gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

#provide encoder q(b|x) gradient by data augmentation
u_noise = tf.random_uniform(shape=[N,b_dim],maxval=1.0) #sample K \uv
P1 = tf.sigmoid(-log_alpha_b)
E1 = tf.cast(u_noise>P1,tf.float32)
P2 = 1 - P1
E2 = tf.cast(u_noise<P2,tf.float32)


F1 = fun1(x_binary,log_alpha_b,E1,axis_dim=1,reuse_encoder=True,reuse_decoder=True)
F2 = fun1(x_binary,log_alpha_b,E2,axis_dim=1,reuse_encoder=True,reuse_decoder=True)

alpha_grads = tf.expand_dims(F1-F2,axis=1)*(u_noise-0.5) #N*d_b
#alpha_grads = tf.reduce_mean(alpha_grads,axis=1) #N*d_b
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is N*d_b, alpha_grads is N*d_b, inf_vars is d_theta
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% TRAIN
# get data
from scipy.io import loadmat
from preprocess import preprocess
train = np.array(loadmat('binarized_mnist_train.amat')['X'])
train_data = preprocess(train)

test = np.array(loadmat('binarized_mnist_test.amat')['X'])
test_data = preprocess(test)

valid = np.array(loadmat('binarized_mnist_valid.amat')['X'])
valid_data = preprocess(valid)


directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
  
batch_size = 50
total_points = train.shape[0]
total_batch = int(total_points / batch_size)
total_test_batch = int(test.shape[0] / batch_size)
total_valid_batch = int(valid.shape[0] / batch_size)

training_epochs = 1200
display_step = total_batch

#%%
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs = data.next_batch(batch_size)  
        cost_eval.append(sess.run(neg_elbo,{x:xs}))
    return np.mean(cost_eval)

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)

np_lr = 0.0001
EXPERIMENT = 'MNIST_Bernoulli_ARM' + '_non_'+str(int(np.random.randint(0,100,1)))
print('Training starts....',EXPERIMENT)
print('Learning rate....',np_lr)

sess=tf.InteractiveSession()
sess.run(init_op)
step = 0

import time
start = time.time()
COUNT=[]; epoch_list=[];TIME=[]
evidence_r = []

for epoch in range(training_epochs):
    
    record = []    
    for i in range(total_batch):
        train_xs = train_data.next_batch(batch_size)  
        _,cost = sess.run([train_op,gen_loss],{x:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
    
    print(epoch,'cost=',np.mean(record),'with std=',np.std(record))
    
    if epoch%1 == 0:
        COUNT.append(step);  TIME.append(time.time()-start)
    if epoch%5 == 0:
        
        avg_evi_val = evidence(sess, valid_data, -neg_elbo, batch_size, S = 100, total_batch=10)
        print(epoch,'The validation NLL is', -np.round(avg_evi_val,2))
        evidence_r.append(np.round(avg_evi_val,2))        
        
        all_ = [COUNT,TIME,evidence_r]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'wb'))  
print(EXPERIMENT)








