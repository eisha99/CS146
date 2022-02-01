#!/usr/bin/env python
# coding: utf-8

# PCW 4.1

# In[1]:


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt


# In[13]:


#source =  https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution
#I will use the following parameters:  mu (μ), either lambda (λ) or nu (ν), alpha (α), beta (β) as defined on the source linked above

def normingamma_pdf(x, sigmaa, mu, nu, alpha, beta):

    #The probability density function of the normal-inverse-gamma distribution at x (mean) and sigma2 (variance).

    return (
        sts.norm.pdf(x, loc=mu, scale=np.sqrt(sigmaa / nu)) *
        sts.invgamma.pdf(sigmaa, a=alpha, scale=beta))
#sigmaa = sigma squared

def normingamma_rvs(mu, nu, alpha, beta, size=1):

    #Generate n samples from the normal-inverse-gamma distribution. This function
    #returns a (size x 2) matrix where each row contains a sample, (x, sigma2).

    # Sample sigmaa (sigma squared) from the inverse-gamma distribution
    sigmaa = sts.invgamma.rvs(a=alpha, scale=beta, size=size)
    # Sample x from the normal distribution
    x = sts.norm.rvs(loc=mu, scale=np.sqrt(sigmaa / nu), size=size)
    return np.vstack((x, sigmaa)).transpose()


# Using methods in the study guide, I obtained the following normal-inverse-gamma prior hyperparameters:
# 
# mu0 = 2.3        
# nu0 = 5.5    
# alpha0 = 5.56
# beta0 = 12.54

# In[16]:


mu0 = 2.3        
nu0 = 5.5    
alpha0 = 5.56
beta0 = 12.54

samples_0 = normingamma_rvs(mu0, nu0, alpha0, beta0, size=12) 

# PLOT THE NORMAL PDF CORRESPONDING TO EACH SAMPLE ABOVE
x = np.linspace(-8, 12, 800)

plt.figure(figsize = [12, 6])

for mu, sigmaa in samples_0:
    plt.plot(x, sts.norm.pdf(x, loc = mu, scale = np.sqrt(sigmaa)), 
             label = f"mu: {mu:.1f}, sd: {np.sqrt(sigmaa):.2f}"
            )
plt.title("Normal Distribution Samples")
plt.xlabel("Mean")
plt.ylabel("Probability")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# 

# In[ ]:




