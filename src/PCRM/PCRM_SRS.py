# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:42:40 2019
@author: YajuanLi

Code for generating experiment results for the portfolioi credit risk model in the paper (Table 1), 
Li, Kaplan, and Nakayama, "Monte Carlo Methods for Economic Capital",
to appear in INFORMS Journal on Computing.

This code is specificly for SRS method for PCRM.

This code is in python language, you can use any Python complier (e.g, Spyder) to run it. 

By default, you don't need to change anything, and directly run the code to get results for SRS.

Since there are random numbers generated in the code, each run of the code may get a slightly different result.

Please look at paper seciton 3. Simple Random Sampling for the SRS method.
Please look at paper section 7.2. Portfolio Credit Risk Model in the paper for details of the model.

"""
import math
import numpy as np
from scipy.stats import norm

m=1000#number of obligors
d=10#number of systematic risk factors
#x=20
replication=100#this is the number of replications we need in total

#print('replication times',replication)
#print('SRS')
#print('m',m)
#print('d',d)
totaln=2000# this is the entire sample size for each replication
n=totaln
batch=10# there will be 10 groups * m=200 samples/group
m1=int(n/batch)#m=200
t_value=2.262# this is t-value foor Confidence interval cauculation


#For this
#model we can compute analytically the mean is 104.02, but this may not be possible for
#more complicated models, and our simulation experiments treat mean as unknown, requiring
#estimation.

# We get the approximate true mean/quantile by using SRS to genearte 10^7 samples and calcuate the mean/quantile.
# You can think "true" mean/quantile are "given" in this code.

true_mean= 104.02482333163012


#quantile choice p= 0.999
pquantile3= 0.999

true_quantile3= 1884.6057631853332

true_eta3= 1780.58093985



pquantile=pquantile3
true_quantile=true_quantile3
true_eta=true_eta3
lpquantile=0.001

def variance(L,mean):#this is function to calculate variance
    suml=0
    for i in range(0,len(L)):
        suml=suml+(L[i]-mean)**2
    avg=suml/(len(L)-1)
    return avg

def readfile():
    #this is for reading data, where the data is pregenerated infomation for a_k in the paper, 
    #which is used later to multiply Z
    all_data=open("a1000d10.txt", 'r')
    data=[]
    for line in all_data:
        row=[float(num)for num in line.split()]
        data.append(row)
    return data

A=readfile()
p=[0.01*(1+math.sin(16*math.pi*k/m)) for k in range(1,m+1)]#Our experiments used p_k as in Glasserman and Li (2005).

beta=[2*(math.ceil(5*k/m))**2 for k in range(1,m+1)]# this is for generating C_k later (the upper threshold of uniform distribution)

inv_p=norm.ppf(p)#this is for generating standard normal random numbers by inversing p
x=[norm.ppf(1-a) for a in p] #for genearing constant w_k in the model (see w_k in the paper)
B=[math.sqrt(1-np.dot(x,y)) for x,y in zip(A,A)]#for calculating b_k in the model (see b_k in the paper)

#below are variables having 1000 numbers of 0 for later use
eta_section=[0]*replication
CI_s_eta=[0]*replication
HW_s_eta=[0]*replication


eta_batch=[0]*replication
CI_b_eta=[0]*replication
HW_b_eta=[0]*replication

#above are variables having 1000 numbers of 0 for later use

quantile_estimator=[0]*batch#variable having 10 numbers of 0 for later use
mean_estimator=[0]*batch#variable having 10 numbers of 0 for later use

L_SRS=[0]*m1#variable having 200 numbers of 0 for later use

L_SRS_B=[0]*n#variable having 2000 numbers of 0 for later use

AZ=[0]*m#variable having 200 numbers of 0 for later use
epsilon=[0]*m#variable having 1000 numbers of 0 for later use
X1=[0]*m#variable having 1000 numbers of 0 for later use
Z=[0]*d#variable having 10 numbers of 0 for later use

for r in range(0,replication):
    cclb=0
    for i in range(0,batch):
        for j in range(0,m1): 
           
            Z=np.random.normal(loc = 0, scale = 1, size =d)#generate Z in the model (see Z in the paper)
            
            epsilon=np.random.normal(loc = 0, scale = 1, size =m)#generate epsilon in the model (see epsilon (in greek letter) in the paper)
            
            X=np.dot(A,Z)+B*epsilon#this is the a_k*Z+b_k*epsilon in the model (see equation (80))
            
            Y=np.greater(X,x)*1# this is the I function in the model (see equation (80)), note that we let S=1, so we don't need to include it here
            c_b=np.random.uniform(0,beta)#this is for generating C_k in the model (the constant LGD in Glasserman and Li (2005))
            C=np.greater(Y,0)*c_b#this is for getting c_k where I funciton is 1
            
            l=sum(C)#this is c(X) in the model(see equation (80))
            L_SRS[j]=l# this is saving the c(X) to the list L_SRS i_th element, recall we pregenerated 200 0s before replication starts.
            L_SRS_B[cclb]=l#also save the c(X) to the larger list
            cclb=cclb+1
        mean_estimator[i]=np.mean(L_SRS)# get the estimator of mean by using build-in mean funciton on the list
        
        L_SRS=sorted(L_SRS)        #sort the list
        s=math.ceil(m1*pquantile)-1 #get the quantile estimator index
       
        quantile_estimator[i]=L_SRS[s]#get quantile estimator
        
    eta_estimator=np.subtract(quantile_estimator,mean_estimator)#get EC estimator
   
    
    #sectioning result
    L_SRS_B=sorted(L_SRS_B) #sort the list
    s=math.ceil(n*pquantile)-1#get the quantile estimator index
    quantile_n=L_SRS_B[s]#get quantile estimator for sectioning
    
    mean_n=np.mean(L_SRS_B)#get mean estimator for sectioning
    eta_n=quantile_n-mean_n#get EC estimator for sectioning
   
    eta_section[r]=eta_n#save EC estimator 
    eta_var=variance(eta_estimator,eta_n)#calculate EC estimators variance
    CI_s_1=eta_n-t_value*math.sqrt(eta_var/batch)#get left side of confidence interval (CI)
    CI_s_2=eta_n+t_value*math.sqrt(eta_var/batch)#get right side of CI
    CI_s_eta[r]=[CI_s_1,CI_s_2]#save the CI
    HW_s_eta[r]=t_value*math.sqrt(eta_var/batch)/eta_n#calcualte the relative half width (RHW) for sectioning
    
    
    
    #batching result
    quantile_n=np.mean(quantile_estimator)#get quantile estimator for batching
    eta_n=quantile_n-mean_n#get EC estimator for batching
    eta_batch[r]=eta_n#save EC estimator 
    eta_var=variance(eta_estimator,eta_n)#calculate EC estimators variance
    CI_b_1=eta_n-t_value*math.sqrt(eta_var/batch)
    CI_b_2=eta_n+t_value*math.sqrt(eta_var/batch)
    CI_b_eta[r]=[CI_b_1,CI_b_2]#save the CI
    HW_b_eta[r]=t_value*math.sqrt(eta_var/batch)/eta_n#calcualte the RHW for batching 
    
   
    
print('################sectioning##############')
#the following 3 lines are for calculating coverage for sectioning
CI_s_eta=np.array(CI_s_eta)
rc = np.sum((CI_s_eta[:,0] <true_eta) & (CI_s_eta[:,1] > true_eta))
coverage_eta=rc/replication

AHW_s_eta=np.mean(HW_s_eta)#calculate ARHW for sectioning

#the following 2 lines are for calculating RRMSE for sectioning
MSE=np.mean(np.square(np.subtract(eta_section,true_eta)))
RRMSE=np.sqrt(MSE)/true_eta

print('coverage_eta',coverage_eta)
print('ARHW_s_eta',AHW_s_eta)
print('RRMSE',RRMSE)
print('-------------------------------------------------------------------')


print('####################batching########################')
#the following 3 lines are for calculating coverage for batching
CI_b_eta=np.array(CI_b_eta)
rc = np.sum((CI_b_eta[:,0] <true_eta) & (CI_b_eta[:,1] > true_eta))
coverage_eta=rc/replication

AHW_b_eta=np.mean(HW_b_eta)#calculate ARHW for batching

#the following 2 lines are for calculating RRMSE for batching
MSE=np.mean(np.square(np.subtract(eta_batch,true_eta)))
RRMSE=np.sqrt(MSE)/true_eta

print('coverage_eta',coverage_eta)
print('ARHW_b_eta',AHW_b_eta)
print('RRMSE',RRMSE)
print('-------------------------------------------------------------------')
