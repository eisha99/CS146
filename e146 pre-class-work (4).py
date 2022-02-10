#!/usr/bin/env python
# coding: utf-8

# # Pre-class work
# Below is the data set from 6 medical trials on the effect of specific allergen immunotherapy (SIT) on eczema patients.
# 
# | Study          | TG improved      | TG not improved   | CG improved    | CG not improved   |
# |:-------------- | --------:| ------:| ------:| ------:|
# | Di Rienzo 2014 | 20       | 3      | 9      | 6      |
# | Galli 1994     | 10       | 6      | 11     | 7      |
# | Kaufman 1974   | 13       | 3      | 4      | 6      |
# | Qin 2014       | 35       | 10     | 21     | 18     |
# | Sanchez 2012   | 22       | 9      | 12     | 17     |
# | Silny 2006     | 7        | 3      | 0      | 10     |
# | **Totals**     | **107**  | **34** | **57** | **64** |
# 
# * TG = Treatment group
# * CG = Control group
# 
# The model we used was that each trial's results were generated from a binomial distribution over the number of improved patients with a common improvement rate parameter shared between all trials.
# 
# For the treatment group we use a subscript $t$:
# 
# $$\begin{align}
# k_{ti} &\sim \text{Binomial}(n_{ti}, p_t) \qquad i=1,2,\ldots 6\\
# p_t &\sim \text{Beta}(\alpha=1, \beta=1)
# \end{align}$$
# 
# For the control group we use a subscript $c$:
# 
# $$\begin{align}
# k_{ci} &\sim \text{Binomial}(n_{ci}, p_c) \qquad i=1,2,\ldots 6\\
# p_c &\sim \text{Beta}(\alpha=1, \beta=1)
# \end{align}$$
# 
# So we have the same model structure for the treatment and control groups, just with different data.
# 
# The code below implements the Stan model for the scenario above.
# 
# * Carefully **read through the code**, including all comments, to understand how Stan is used to represent the medical trial model.
# * **Run the code** to see inference results for the treatment group.
# * **Complete the two tasks** at the end of the notebook.

# In[1]:


import pystan

import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# For Stan we provide all known quantities as data, namely the observed data
# and our prior hyperparameters.
eczema_data = {
    'treatment': {
        'alpha': 1,  # fixed prior hyperparameters for the
        'beta': 1,   # beta distribution
        'num_trials': 6,  # number of trials in the data set
        'patients': [23, 16, 16, 45, 31, 10],  # number of patients per trial
        'improved': [20, 10, 13, 35, 22, 7]},  # number of improved patients per trial
    'control': {
        'alpha': 1,
        'beta': 1,
        'num_trials': 6,
        'patients': [15, 18, 10, 39, 29, 10],
        'improved': [9, 11, 4, 21, 12, 0]}}


# In[ ]:


# Below is the Stan code for the medical trial data set. Note that the Stan
# code is a string that is passed to the StanModel object below.

# We have to tell Stan what data to expect, what our parameters are and what
# the likelihood and prior are. Since the posterior is just proportional to
# the product of the likelihood and the prior, we don't distinguish between
# them explicitly in the model below. Every distribution we specify is
# automatically incorporated into the product of likelihood * prior.

stan_code = """

// The data block contains all known quantities - typically the observed
// data and any constant hyperparameters.
data {  
    int<lower=1> num_trials;  // number of trials in the data set
    int<lower=0> patients[num_trials];  // number of patients per trial
    int<lower=0> improved[num_trials];  // number of improved patients per trial
    real<lower=0> alpha;  // fixed prior hyperparameter
    real<lower=0> beta;   // fixed prior hyperparameter
}

// The parameters block contains all unknown quantities - typically the
// parameters of the model. Stan will generate samples from the posterior
// distributions over all parameters.
parameters {
    real<lower=0,upper=1> p;  // probability of improvement - the
                              // parameter of the binomial likelihood
}

// The model block contains all probability distributions in the model.
// This of this as specifying the generative model for the scenario.
model {
    p ~ beta(alpha, beta);  // prior over p
    for(i in 1:num_trials) {
        improved[i] ~ binomial(patients[i], p);  // likelihood function
    }
}

"""


# In[ ]:


# This cell takes a while to run. Compiling a Stan model will feel slow even
# on simple models, but it isn't much slower for really complex models. Stan
# is translating the model specified above to C++ code and compiling the C++
# code to a binary that it can executed. The advantage is that the model needs
# to be compiled only once. Once that is done, the same code can be reused
# to generate samples for different data sets really quickly.

stan_model = pystan.StanModel(model_code=stan_code)


# In[ ]:


# Fit the model to the data. This will generate samples from the posterior over
# all parameters of the model. We start by computing posteriors for the treatment
# data.

stan_results = stan_model.sampling(data=eczema_data['treatment'])


# In[ ]:


# Print out the mean, standard deviation and quantiles of all parameters.
# These are approximate values derived from the samples generated by Stan.
# You can ignore the "lp__" row for now. Pay attention to the row for
# the "p" parameter of the model.
#
# The columns in the summary are
#
#  * mean: The expected value of the posterior over the parameter
#  * se_mean: The estimated error in the posterior mean
#  * sd: The standard deviation of the posterior over the parameter
#  * 2.5%, etc.: Percentiles of the posterior over the parameter
#  * n_eff: The effective number of samples generated by Stan. The
#           larger this value, the better.
#  * Rhat: An estimate of the quality of the samples. This should be
#          close to 1.0, otherwise there might be a problem with the
#          convergence of the sampler.

print(stan_results)


# In[ ]:


# Specify which parameters you want to see in the summary table using
# the "pars" keyword argument. Specify which percentiles you want to
# see using the "probs" keyword argument.
#
# The statement below shows only the 2.5, 50, 97.5 percentiles for the
# parameter p.

print(stan_results.stansummary(pars=['p'], probs=[0.025, 0.5, 0.975]))


# In[ ]:


# Finally, we can extract the samples generated by Stan so that we
# can plot them or calculate any other functions or expected values
# we might be interested in.

posterior_samples = stan_results.extract()
plt.hist(posterior_samples['p'], bins=50, density=True)
plt.title('Sampled posterior probability density for p')
print(
    "Posterior 95% confidence interval for p:",
    np.percentile(posterior_samples['p'], [2.5, 97.5]))
plt.show()


# ## Task 1
# * Reuse the code above to calculate the posterior 95% confidence interval for the probability of improvement in the **control group**.
# * Plot the posterior histograms of the probability of improvement in the treatment and control groups on the same figure.

# In[2]:


#coded in cocalc and transfered to jupyter ro download
stan_results1 = stan_model.sampling(data=eczema_data['control'])
print(stan_results1.stansummary(pars=['p'], probs=[0.025, 0.5, 0.975]))


# plotting hist graph
postsamples = stan_results_c.extract()
plt.hist(postsamples['p'], bins=50, density=True)
plt.title('Sampled posterior probability density for p (\'control group\')')
plt.show()


# In[ ]:


# printing 2 plots
plt.hist(tpostsamples['p'], bins=50, density=True,
        alpha=0.5, label='/Treatment')
plt.hist(cpostsample['p'], bins=50, density=True,
        alpha=0.5, label='/Control')
plt.title('Sampled posterior probability density for p')
plt.xlabel('p')
plt.ylabel('PDF')
plt.legend()
plt.show()


# ## Task 2
# * Using the samples from the treatment and control group posteriors, estimate the probability that treatment is at least 19% (in absolute terms) better than control, $P(p_t > p_c + 0.19)$. We computed this result in Session 3.2 where we solved the same model analytically using the algebra of conjugate distributions.

# In[ ]:


#using both treatment and control
diff1 = np.random.choice(tpostsamples['p'], size=10000) - np.random.choice(tpostsamples['p'], size=10000)
plt.hist(diff1, density=True)

