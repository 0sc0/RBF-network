# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:58:50 2016
This file is used to fit the curve of traffic.
@author: Hansyang
"""
import numpy as np
import matplotlib.pyplot as plt
def logsig(x):
    return 1/(1+np.exp(-x))

#Original Data
#Input: the pupulation， number of vehicle, roadarea from 1990-2009
population=[20.55,22.44,25.37,27.13,29.45,30.10,30.96,34.06,36.42,38.09,39.13,39.99,41.93,44.59,47.30,52.89,55.73,56.76,59.17,60.63]
vehicle=[0.6,0.75,0.85,0.9,1.05,1.35,1.45,1.6,1.7,1.85,2.15,2.2,2.25,2.35,2.5,2.6,2.7,2.85,2.95,3.1]
roadarea=[0.09,0.11,0.11,0.14,0.20,0.23,0.23,0.32,0.32,0.34,0.36,0.36,0.38,0.49,0.56,0.59,0.59,0.67,0.69,0.79]
#Output
passengertraffic=[5126,6217,7730,9145,10460,11387,12353,15750,18304,19836,21024,19490,20433,22598,25107,33442,36836,40548,42927,43462]
freighttraffic=[1237,1379,1385,1399,1663,1714,1834,4322,8132,8936,11099,11203,10524,11115,13320,16762,18673,20724,20803,21804]

# normalize the original data and add the noise
samplein = np.mat([population,vehicle,roadarea]) #3*20
sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()#3*2
sampleout = np.mat([passengertraffic,freighttraffic])#2*20
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()#2*2
sampleinnorm = (2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.transpose()[1]-sampleinminmax.transpose()[0])-1).transpose()
sampleoutnorm = (2*(np.array(sampleout.T).astype(float)-sampleoutminmax.transpose()[0])/(sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])-1).transpose()

#initial the parameters
maxepochs =1000
learnrate = 0.035
errorfinal = 0.5*10**(-3)
samnum = 20
indim = 3
outdim = 2
hiddenunitnum = 8
w1 = 2*np.random.rand(hiddenunitnum,indim)-1
b1 = 2*np.random.rand(hiddenunitnum,1)-1
w2 = 2*np.random.rand(outdim,hiddenunitnum)-1
b2 = 2*np.random.rand(outdim,1)-1
errhistory = []
for i in range(maxepochs):
    hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
    networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    err = sampleoutnorm - networkout
    sse = sum(sum(err**2))
    #Use the err of the whole dataset as the err, rather than one subject, aiming of reduce the err fast
    errhistory.append(sse)
    if sse < errorfinal:
        break
    delta2 = err
    delta1 = np.dot(w2.transpose(),delta2)*hiddenout*(1-hiddenout)
    dw2 = np.dot(delta2,hiddenout.transpose())
    db2 = np.dot(delta2,np.ones((samnum,1)))
    dw1 = np.dot(delta1,sampleinnorm.transpose())
    db1 = np.dot(delta1,np.ones((samnum,1)))
    w2 += learnrate*dw2
    b2 += learnrate*db2
    w1 += learnrate*dw1
    b1 += learnrate*db1
 #For there was a normalization, cacalute the original output use the min and max value
hiddenout = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
networkout2 = (networkout+1)/2
networkout2[0] = networkout2[0]*diff[0]+sampleoutminmax[0][0]
networkout2[1] = networkout2[1]*diff[1]+sampleoutminmax[1][0]
sampleout = np.array(sampleout)

#show the err curve and the results
plt.figure(1)
plt.plot(errhistory,label="error")
plt.legend(loc='upper left')

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(sampleout[0],color="blue", linewidth=1.5, linestyle="-", label="real curve of passengertraffic")
plt.plot(networkout2[0],color="red", linewidth=1.5, linestyle="--",  label="fitting curve")
plt.legend(loc='upper left')
plt.show()
plt.subplot(2,1,2)
plt.plot(sampleout[1],color="blue", linewidth=1.5, linestyle="-", label="real curve of freighttraffic")
plt.plot(networkout2[1],color="red", linewidth=1.5, linestyle="--",  label="fitting curve")
plt.legend(loc='upper left')
plt.show()