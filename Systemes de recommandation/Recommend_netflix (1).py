#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:07:49 2018

@author: bmoos
"""

import scipy.io as sio



train = sio.loadmat('D:/Netflix/netflix_data_app.mat')
train=train['netflix_data_app']
test = sio.loadmat('D:/Netflix/netflix_data_probe.mat')
test=test['netflix_data_probe']

# 480 000 films et 18 000 spectateurs

#%%

#from scipy.io.matlab import mio

import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix


shape=train.shape
zapp=csc_matrix(train,shape)
zte=csc_matrix(test,shape)

[i_app,j_app,v_app]=sparse.find(zapp)# index de app non nuls
[i_val,j_val,v_val]=sparse.find(zte)# index de test non nuls

mean=np.mean(v_app)
# MEMORY ERREUR à ce stade
#%%
Mask= zapp>0.5
Maskt= zte>0.5
n_ratings=zte.count_nonzero()

vec_notes=mean*np.ones(n_ratings)

pred=csc_matrix((vec_notes,(i_val,j_val)),shape=test.shape)

errr = pred-zte

print("Erreur sur le Probe :{}".format(np.sum(errr.power(2))/n_ratings))



#%%Moyenne des films

#t0=time()
zapp=zapp-mean*Mask


mean_films=zapp.sum(axis=0)/Mask.sum(axis=0)

vec_notes=mean*np.ones(n_ratings)

zte=zte-Maskt.multiply(mean_films)
err1=np.sum(zte.power(2))/n_ratings

#t1=time()


#print("erreur : {0:.3f}".format(err1))



#%% Moyenne des spectateurs


#t0=time()


zapp=zapp-Mask.multiply(mean_films)



mean_spec=np.sum(zapp,axis=1)/np.sum(zapp>0.5,axis=1)

zte=zte-Maskt.multiply(mean_spec)
err2=np.sum(zte.power(2))/n_ratings

#t1=time()


print("erreur : {0:.3f}".format(err2))



#%% SVD


Merr=0*Mt

U=U[:,::-1]
D=D[::-1]
Vt=Vt[::-1,:]

p,n=zte.shape
Vt=np.dot(np.diag(D),Vt)



for j in range(0,p):
    Vp=U[j,:]*Maskt[j,:].multiply(Vt)
    Merr[j,:]=csc_matrix(Vp)-Mt[j,:]
    











