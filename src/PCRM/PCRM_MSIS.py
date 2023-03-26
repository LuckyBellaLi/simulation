# -*- coding: utf-8 -*-
"""
Created on Tue Dec  24 16:22:42 2019

@author: Liyajuan

Code for generating experiment results for the portfolioi credit risk model in the paper, 
Li, Kaplan, and Nakayama, "Monte Carlo Methods for Economic Capital",
to appear in INFORMS Journal on Computing.

This code is specificly for MSIS method for PCRM.

This code is in python language, you can use any Python complier (e.g, Spyder) to run it. 

By default, you don't need to change anything, and directly run the code to get results for MSIS.

Since there are random numbers generated in the code, each run of the code may get a slightly different result.

Please look at paper seciton 5.1. Measure-Speci c Importance Sampling (MSIS) for the MSIS method.
Please look at paper section 7.2. Portfolio Credit Risk Model in the paper for details of the model. 
Please look at paper Appendix J: Two-Step IS to Estimate Extreme Quantile and EC in PCRM with
Random Loss Given Default for more details of Two-step IS
"""
import math
import random
import scipy
import numpy as np
from scipy.stats import norm

from scipy.optimize import minimize

m=1000
d=10
replication=1000
t_value=2.262
totaln=1500## this is the sample size to generate estimators for each replication, 
#and we need another 500 samples to do quantile approximation, so in total sample size is 1500+500=2000
n=int(totaln/2)# 750 for IS method to generate quantile estimator, 750 for SRS method to generate mean estimator
n1=n
n2=n
batch=10
m1=int(n1/batch)# there will be 10 groups,  m1=75 samples/group
m2=m1
print('IS')
print('m',m)
print('d',d)

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

delta=0.98
p=np.array([0.01*(1+math.sin(16*math.pi*k/m)) for k in range(1,m+1)])
beta=np.array([2*(math.ceil(5*k/m))**2 for k in range(1,m+1)])


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

print('true_quantile',true_quantile)
print('true_eta',true_eta)
print('1-pquantile',lpquantile)



p_level=pquantile
lp_level=lpquantile
n_QA=100
C=[0]*m
Y=[0]*m
rk=[0]*m



L_QA=[0]*n_QA
likelihood_QA=[0]*n_QA
S=np.sum(beta)


print('delta = ',delta)
rx=5
print('number of x is: ',rx)
x_quantile=(1-delta**np.array(range(1,rx+1)))*S#this is for crude approximation to the quantile
#see paper J.3. Two-step IS to Estimate Quantile for details


print('x_quantile method is (1-delta**k)*S')
print('x_quantile',x_quantile)
tail_probability=[0]*rx


inv_p=norm.ppf(p)
xx = [norm.ppf(1-a) for a in p]


B=[math.sqrt(1-np.dot(x,y)) for x,y in zip(A,A)]


beta_square=np.square(beta)

#this function is for findng optimal mean for generating Z later
#see paper J.2.2. Applying Two-Step IS to Estimate P(Y > x) for more details
def optimal_z(x_ini):
    Z=x_ini
   
    pkz=norm.cdf((np.dot(A,Z)+inv_p)/B)
    
    dot=np.dot(beta,pkz)
   
    numrator=x-dot/2
    pkz_square=np.square(pkz)
    sum_denominator=np.dot(beta_square,pkz)/3-(np.dot(beta_square,pkz_square))/4
    denominator=math.sqrt(sum_denominator)
    normcdf=norm.cdf(numrator/denominator)
    
    result=(1-normcdf)*math.exp(-np.dot(Z,Z)/2)
    return -result

ini_point_numbers=10



def variance(L,mean):#this is function to calculate variance
    suml=0
    for i in range(0,len(L)):
        suml=suml+(L[i]-mean)**2
    avg=suml/(len(L)-1)
    return avg

def mgf(theta):#conditional moment generating funciton
    if theta==0:
        return 1
    pk=np.array(pkz)
    bet=np.array(beta)
    th_be=theta*bet
    exp=np.exp(th_be)
    sumlist=np.prod(1-pk+pk*(exp-1)/th_be)
    return sumlist

def derivative_cgf(theta):#return derivative of conditional cgf
    if theta==0:
        return np.dot(pkz,beta)/2
    pk1=1/pkz
    th_be=beta*theta
    th_be_square=th_be*theta
    exp=np.exp(th_be)
    mk=(exp-1)/th_be
    ak=1/(pk1+(mk-1))
    bk=((th_be-1)*exp+1)/th_be_square
    akbk=np.dot(ak,bk)
    sumlist=np.sum(akbk)
    return sumlist  
def theta_rootfinder(theta):#this function is for finding root theta in J.2.2. Applying Two-Step IS to Estimate P(Y > x)
    return derivative_cgf(theta)-x

eta_section=[0]*replication
CI_s_eta=[0]*replication
HW_s_eta=[0]*replication
quantile_section=[0]*replication
CI_s_quantile=[0]*replication
HW_s_quantile=[0]*replication
mean_section=[0]*replication
CI_s_mean=[0]*replication
HW_s_mean=[0]*replication

eta_batch=[0]*replication
CI_b_eta=[0]*replication
HW_b_eta=[0]*replication
quantile_batch=[0]*replication
CI_b_quantile=[0]*replication
HW_b_quantile=[0]*replication
mean_batch=[0]*replication
CI_b_mean=[0]*replication
HW_b_mean=[0]*replication

quantile_estimator=[0]*batch
mean_estimator=[0]*batch
eta_estimator=[0]*batch

L=[0]*m1
L_SRS=[0]*m2
likelihood=[0]*m1

LB=[0]*n1
Blikelihood=[0]*n1
L_SRS_B=[0]*n2

my_keys=["x","fun"]
for r in range(0,replication):   
    for r_QA in range(0,rx):  #this loop is for getting crude quantile approximation
    #see details at paper J.3. Two-step IS to Estimate Quantile
        x=x_quantile[r_QA]
       
        mu=minimize(optimal_z,x0=np.random.uniform(0,5,d),method='COBYLA')["x"]#findng optimal mean for generating Z later
        
        for j in range(0,n_QA):
            #generate Z with mean mu
            
            Z=np.random.normal(loc = mu, scale = 1, size =d)
            pkz=norm.cdf((np.dot(A,Z)+inv_p)/B)
            E=np.dot(pkz,beta)/2
           
            if  E>=x:
                theta=0
            else:               
                a=0
                if derivative_cgf(a)-x>=0:
                    b=-0.1
                    while(derivative_cgf(b)-x>0):
                        b=2*b
                else:
                    b=0.1
                    while(derivative_cgf(b)-x<=0):
                        b=2*b
               
                theta=scipy.optimize.ridder(theta_rootfinder, a, b)
            
            rk=pkz if theta==0 else pkz*(np.exp(theta*beta)-1)/((1-pkz)*theta*beta+(pkz*(np.exp(theta*beta)-1)))
            uy=np.random.uniform(0,1,m)
            Y=(uy<rk)*1
            
            c_b=np.random.uniform(0,beta)
            uc=np.random.uniform(0,1,m)
            C=np.greater(Y,0)*c_b if theta==0 else np.greater(Y,0)*1/theta*np.log(1+(np.exp(theta*beta)-1)*uc)
            
            #compute c(X)
            l=sum(C)
            L_QA[j]=l
            likeli=math.exp(-theta*l)*mgf(theta)*math.exp(-np.dot(mu,Z)+np.dot(mu,mu)/2)
            likelihood_QA[j]=likeli
        tailp=np.greater(L_QA,x)*likelihood_QA
        tail_probability[r_QA]=np.mean(tailp)
   
    curr_quantile=0
    for i in range(0,len(tail_probability)):
        if i!=len(tail_probability)-1:
            if tail_probability[i]>lp_level and tail_probability[i+1]<lp_level:
                l_k=(x_quantile[i]-x_quantile[i+1])/(math.log10(tail_probability[i])-math.log10(tail_probability[i+1]))
                curr_quantile=x_quantile[i+1]+l_k*(math.log10(lp_level)-math.log10(tail_probability[i+1]))
                break
    if curr_quantile==0:
        if lp_level>tail_probability[0]:
            curr_quantile=x_quantile[0]
        if lp_level<tail_probability[-1]:
            curr_quantile=x_quantile[-1]
    quantile_approximation=curr_quantile
    x=quantile_approximation
    #quantile_approximation ends
    
    join_mu_fun=[[minimize(optimal_z,x0=[random.uniform(0,5) for k in range(d)],method='COBYLA')[k] for k in my_keys] for i in range(ini_point_numbers)]
    mu_candidate,candidate_result=zip(*join_mu_fun)
    mu_index=candidate_result.index(min(candidate_result))
    mu=mu_candidate[mu_index] #get mean for generating Z
    
    cclb1=0
    cclb2=0
    
    for i in range(0,batch):
        for j in range(0,m1):    
            #generate Z with mean mu
            Z=np.random.normal(loc = mu, scale = 1, size =d)
           
            pkz=np.array(norm.cdf((np.dot(A,Z)+inv_p)/B))
            
            E=np.dot(pkz,beta)/2
           
            if  E>=x:
                theta=0
            else:               
                a=0.01
                if derivative_cgf(a)-x>0:
                    b=a-1
                    while (derivative_cgf(b)-x>0):
                        b=b-0.05
                else:
                    b=a+1
                    while (derivative_cgf(b)-x<=0):
                        b=b+0.05
                
                theta=scipy.optimize.ridder(theta_rootfinder, a, b)
            
            rk=pkz if theta==0 else pkz*(np.exp(theta*beta)-1)/((1-pkz)*theta*beta+(pkz*(np.exp(theta*beta)-1)))
            
            uy=np.random.uniform(0,1,m)
            Y=(uy<rk)*1
            
            c_b=np.random.uniform(0,beta)
            uc=np.random.uniform(0,1,m)
            C=np.greater(Y,0)*c_b if theta==0 else np.greater(Y,0)*1/theta*np.log(1+(np.exp(theta*beta)-1)*uc)
            #compute c(X)
            l=sum(C)
            L[j]=l
            LB[cclb1]=l
            #compute 2-step likelihood ratio
            likeli=math.exp(-theta*l)*mgf(theta)*math.exp(-np.dot(mu,Z)+np.dot(mu,mu)/2)
            
            likelihood[j]=likeli
            Blikelihood[cclb1]=likeli
            cclb1=cclb1+1
            
        LR=np.column_stack((L,likelihood))
        LR=LR[LR[:,0].argsort()]
       
        sumR=0
        for k in range(m1-1,-1,-1):
            sumR=sumR+LR[k][1]
            if sumR>lpquantile*m1:
                break
        quantile_estimator[i]=LR[k][0]# get quantile_estimator for this batch
        
        for j in range(0,m2):
           
            Z=np.random.normal(loc = 0, scale = 1, size =d)
           
            epsilon=np.random.normal(loc = 0, scale = 1, size =m)
            
            X=np.dot(A,Z)+B*epsilon
           
            Y=np.greater(X,xx)*1
            c_b=np.random.uniform(0,beta)
            C=np.greater(Y,0)*c_b
            
            l_SRS=sum(C)
            L_SRS[j]=l_SRS
            L_SRS_B[cclb2]=l_SRS
            cclb2=cclb2+1
        mean_estimator[i]=np.mean(L_SRS)## get mean_estimator for this batch 
    eta_estimator=np.subtract(quantile_estimator,mean_estimator)## get eta_estimator for this batch
    
    #sectioning
    LBR=np.column_stack((LB,Blikelihood))
    LBR=LBR[LBR[:,0].argsort()]
    sumR=0
    for k in range(n-1,-1,-1):
          sumR=sumR+LBR[k][1]
          if sumR>=lpquantile*n:
              break
    quantile_n=LBR[k][0]## get quantile_estimator for sectioning
    
    mean_n=np.mean(mean_estimator)## get mean_estimator for sectioning
    eta_n=quantile_n-mean_n## get eta_estimator for sectioning
    #construct CI
    eta_section[r]=eta_n
    eta_var=variance(eta_estimator,eta_n)
    CI_s_1=eta_n-t_value*math.sqrt(eta_var/batch)
    CI_s_2=eta_n+t_value*math.sqrt(eta_var/batch)
    CI_s_eta[r]=[CI_s_1,CI_s_2]#save CI for sectioning
    HW_s_eta[r]=t_value*math.sqrt(eta_var/batch)/eta_n#calcualte the relative half width (RHW) for sectioning 
    
    
    #batching
    quantile_n=np.mean(quantile_estimator)## get quantile_estimator for batching
    eta_n=quantile_n-mean_n## get eta_estimator for batching
    #construct CI
    eta_batch[r]=eta_n
    eta_var=variance(eta_estimator,eta_n)
    CI_b_1=eta_n-t_value*math.sqrt(eta_var/batch)
    CI_b_2=eta_n+t_value*math.sqrt(eta_var/batch)
    CI_b_eta[r]=[CI_b_1,CI_b_2]#save CI for batching
    HW_b_eta[r]=t_value*math.sqrt(eta_var/batch)/eta_n#calcualte the relative half width (RHW) for batching
    
    
print('replication times',replication)
print('total n',totaln)
print('pquantile',pquantile)
print('-------------------------------------------------------------------')

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
#the following 2 lines are for calculating RRMSE for sectioning
MSE=np.mean(np.square(np.subtract(eta_batch,true_eta)))
RRMSE=np.sqrt(MSE)/true_eta
print('coverage_eta',coverage_eta)
print('ARHW_b_eta',AHW_b_eta)
print('RRMSE',RRMSE)
print('-------------------------------------------------------------------')

