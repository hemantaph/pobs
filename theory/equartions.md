# Posterior overlap

## Contents

- [Posterior overlap](#posterior-overlap)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Posterior Overlap and Hypothesis Testing in Lensed Gravitational Wave Astronomy](#posterior-overlap-and-hypothesis-testing-in-lensed-gravitational-wave-astronomy)
  - [Initial setup](#initial-setup)
    - [Odds ratio calculation](#odds-ratio-calculation)
    - [Bayes factor calculation](#bayes-factor-calculation)
    - [Parameter space](#parameter-space)
    - [Factoring out the PE Prior](#factoring-out-the-pe-prior)
  - [Re-parameterization](#re-parameterization)
    - [Concept of re-parameterization](#concept-of-re-parameterization)
    - [Part 1: Time](#part-1-time)
    - [Part 2: Distance](#part-2-distance)
    - [Part 3: KDE scaling](#part-3-kde-scaling)
  - [Renormalization](#renormalization)
    - [Steps for Normalizing KDE](#steps-for-normalizing-kde)
  - [Marginal likelihood for Lensed hypothesis](#marginal-likelihood-for-lensed-hypothesis)
    - [Astrophysical prior of lensed events](#astrophysical-prior-of-lensed-events)
    - [Posterior distributions](#posterior-distributions)
      - [First event](#first-event)
      - [Second event](#second-event)
    - [Astrophysical prior of unlensed events](#astrophysical-prior-of-unlensed-events)
  - [Numerical method](#numerical-method)
    - [Introduction Monte-Carlo Integration](#introduction-monte-carlo-integration)
    - [Sampling in the parameter space](#sampling-in-the-parameter-space)
    - [Bayes factor, analytical form](#bayes-factor-analytical-form)
      - [Numerator](#numerator)
      - [Denominator](#denominator)
    - [Bayes factor, numerical integration](#bayes-factor-numerical-integration)
  - [Extending posterior overlap to four images](#extending-posterior-overlap-to-four-images)
    - [Bayes factor, analytical form](#bayes-factor-analytical-form-1)
      - [Numerator](#numerator-1)
      - [Denominator](#denominator-1)

## Introduction

### Posterior Overlap and Hypothesis Testing in Lensed Gravitational Wave Astronomy 

Gravitational wave (GW) astronomy has emerged as a pivotal field in understanding the cosmos, particularly through the observation of events such as compact binary mergers. A major scientific challenge within this domain is distinguishing between lensed and unlensed gravitational wave events. Lensed events occur when a massive body, like a galaxy or a cluster of galaxies, bends the path of the gravitational waves emanating from a distant astronomical event, such as a binary merger, causing multiple detectable signals from what is essentially a single event. Conversely, unlensed events involve signals that reach observers without such interference. This distinction is crucial for accurate cosmological measurements and understanding the distribution of matter in the universe.

Despite advancements in detection and analysis techniques, there are significant gaps in methodologies used to verify if detected signals are lensed. Previous studies relied on comparing the posterior probability distributions of observed event parameters to determine their lensing probability. However, these analyses often used outdated or oversimplified astrophysical models or no moddels at all, and lacked in mathematical rigor, failing to incorporate complex processes such as reparametrization, Jacobians, or renormalization of the parameter space. Additionally, prior methods such as employing scipy's multi-dimensional gaussian_kde did not adequately capture the necessary parameter correlations, often leading to unreliable probability density functions (PDFs). This was primarily due to the method's approach of fitting multi-dimensional spheres with a constant bandwidth at each parameter point.

Our study addresses these deficiencies by implementing an advanced Bayesian analysis framework that integrates model-dependent astrophysical priors reflecting the latest understanding of both lensed and unlensed events. We also employ sophisticated mathematical techniques including reparametrization, Jacobian transformations, and renormalization to enhance the accuracy of our Bayesian framework. Instead of the traditional gaussian_kde, we utilize sklearn's BayesianGaussianMixture model, which provides a more accurate modeling of parameter correlations and captures the complex differences expected between the parameter distributions of lensed and unlensed events.

By addressing the limitations of previous studies, our research provides a robust tool for interpreting the intricate nature of gravitational wave observations. This refined approach significantly improves the estimation of the odds ratio for the lensing hypothesis and enhances the reliability of the hypothesis testing process.

## Initial setup 

### Odds ratio calculation

To analyze posterior overlap, consider two gravitational wave (GW) events. The posterior distributions of the parameters for the first and second events are denoted as $P(\theta_1|d_1)$ and $P(\theta_2|d_2)$, respectively. To test if these events are images of the same astrophysical source, we evaluate the hypothesis $H_L$ (lensed) against $H_U$ (unlensed). The odds ratio is employed to quantify the evidence supporting the lensed hypothesis. The probability of the hypothesis $H_L$ being correct given the data from the two events is $P(H_L|{d_1,d_2})$, and for $H_U$ it is $P(H_U|{d_1,d_2})$. The odds ratio is defined as:

$$
\begin{align}
\mathcal{O}_U^L &= \frac{P(H_L|\{d_1,d_2\})}{P(H_U|\{d_1,d_2\})} \\ 
&= \frac{P(\{d_1,d_2\}|H_L)P(H_L)}{P(\{d_1,d_2\})} \frac{P(\{d_1,d_2\})}{P(\{d_1,d_2\}|H_U)P(H_U)} \\
&= \frac{P(\{d_1,d_2\}|H_L)}{P(\{d_1,d_2\}|H_U)}\frac{P(H_L)}{P(H_U)} \\
&= \mathcal{B}_U^L \mathcal{P}_U^L \\ 
\end{align}
$$

where $\mathcal{B}_U^L$ is the Bayes factor and $\mathcal{P}_U^L$ the prior odds. Using Bayes' theorem, the odds ratio becomes the product of these two terms. 

We assume $\mathcal{P}_U^L = 1$ to maintain neutrality between the lensed and unlensed hypotheses, focusing on the influence of the data via the Bayes factor $\mathcal{B}_U^L$ in the odds ratio calculation.

### Bayes factor calculation

The Bayes factor, representing the ratio of the marginal likelihoods of the data under the two hypotheses, is calculated as follows:

$$
\begin{align}
\mathcal{B}_U^L &= \frac{P(\{d_1,d_2\}|H_L)}{P(\{d_1,d_2\}|H_U)} \\
&= \frac{\int d\theta P(\{d_1,d_2\}|\theta,H_L)P(\theta|H_L)}{\int d\theta P(\{d_1,d_2\}|H_U) P(\theta|H_U)} \\
\end{align}
$$

Assuming the likelihood functions $P({d_1,d_2}|\theta,H_L)$ and $P({d_1,d_2}|H_U)$ are consistent across both events, the integral simplifies to:

$$
\begin{align}
\mathcal{B}_U^L &= \frac{ \int d\theta \frac{P(\theta|d_1) \cancel{P(d_1)}}{P_{pe}(\theta)} \frac{P(\theta|d_2) \cancel{P(d_2)}}{P_{pe}(\theta)} P_{astro}(\theta|H_L) }
{\int d\theta \frac{P(\theta|d_1) \cancel{P(d_1)}}{P_{pe}(\theta)} \frac{P(\theta|d_2) \cancel{P(d_2)}}{P_{pe}(\theta)} P_{astro}(\theta|H_U)} \\
&=  \frac{ \int d\theta \frac{P(\theta|d_1)}{P_{pe}(\theta)} \frac{P(\theta|d_2)}{P_{pe}(\theta)} P_{astro}(\theta|H_L) }
{\int d\theta \frac{P(\theta|d_1)}{P_{pe}(\theta)} \frac{P(\theta|d_2)}{P_{pe}(\theta)} P_{astro}(\theta|H_U)} \\
\end{align}
$$

The subscript 'astro' indicates that the prior distribution of the parameters reflects the astrophysical context. The prior distribution $P_{pe}(\theta)$ is utilized in parameter estimation for the events. The posterior distributions given the data from the first and second GW events are $P(\theta|d_1)$ and $P(\theta|d_2)$, respectively. The integrands for event 1 and event 2 are combined for mathematical convenience, and to utilize the time delay information in both the numerator and denominator.

### Parameter space

For clarity in the analysis, the intrinsic and extrinsic parameters for the compact binary events are listed below:


| Parameter || Description | Unit|
|:---:|---|:---:|---|
| Event 1 | Event 2 | ||
| $m_{1,1}$ | $m_{1,2}$ | mass of the heavier BH | $\mathcal{M}_{\odot}$ | 
| $m_{2,1}$ | $m_{2,2}$ | mass of the lighter BH | $\mathcal{M}_{\odot}$ |
| $d_{L,1}$ | $d_{L,2}$ | luminosity distance | Mpc |
| $\iota_1$ | $\iota_2$ | inclination angle | rad |
| $t_1$ | $t_2$ | coalescence time | s |
| $\alpha_1$ | $\alpha_2$ | right ascension | rad |
| $\delta_1$ | $\delta_2$ | declination | rad |


### Factoring out the PE Prior

To simplify the integral further, I will make a bold but reasonable assumption that changing the prior distribution of the parameters $P_{pe}(\theta)$ will not affect the posterior distribution of the parameters $P(\theta|d_1)$ and $P(\theta|d_2)$ as long as it is wide enough to cover the parameter distribution and is not biased towards any particular region of the parameter space. With this claim in mind, $P_{pe}(\theta)$ is replaced with multidimensional Uniform Distribution, $P_o(\theta)=1/(\theta_{\text{max}}-\theta_{\text{min}})$. Thus, it can be factored out of the integral and cancels out in the Bayes factor calculation. Now the Bayes factor becomes:

$$
\begin{align}
\mathcal{B}_U^L =  \frac{ \int d\theta P(\theta|d_1) P(\theta|d_2) P_{astro}(\theta|H_L) }
{\int d\theta P(\theta|d_1) P(\theta|d_2) P_{astro}(\theta|H_U)} \\
\end{align}
$$

## Re-parameterization

### Concept of re-parameterization

Proper re-parameterization is necessary if we want to get a bayes-factor value that is comparable to that of the joint parameter estimation of two events under lensing hypothesis. I am addressing two main issues in re-parameterization:

* How to re-scaled the parameter volume?
* How to interpret the use of KDE in numerical method. 

### Part 1: Time

Consider the following integral where the integrand $P(t_1,t_2, \xi)$ is a multi-dimensional joint probability distribution. $\xi$, $\xi_1$ and $\xi_2$ represents the remaining parameters in the respective probability distribution, and $\xi_1,\xi_2\in \xi$. It is useful to point out the various correlation between the parameters in the probability distribution.

* $t_1$ is independent of $t_2$, $\Delta t$ and $\xi$. Parameters $\Delta t$ and $\xi$ are correlated, while $\Delta t$ and $t_2$ are dependent on $t_1$ and to each other.
* $H\in \{H_L, H_U\}$ is the hypothesis that the two events are lensed or unlensed.

$$
\begin{align}
\text{I} &= \int P(t_1,\xi_1|d_1) P(t_2,\xi_2|d_2) P(t_1,t_2,\xi|H) \, dt_1 dt_2 d\xi \\
&= \int \text{F}(t_1,t_2,\xi|d_1, d_2, H) \, dt_1 dt_2 d\xi \\
\end{align}
$$

Where $\text{F}(t_1,t_2,\xi|d_1, d_2, H)$ is some function. We will reparameterise the variable in the probability distribution as $t_2\rightarrow \Delta t -t_1$, with the Jacobian matrix determinant taken into consideration. Jacobian matrix determinant is the scaling factor for the differential element.

**Jacobian matrix determinant**

Given the variables $t_1$, $t_2$, and $\xi$, and introducing $\Delta t$ such that $t_2 = \Delta t + t_1$:

- Define $u = t_1$ (unchanged)
- Define $v = \Delta t$ (where $t_2 = t_1 + \Delta t$)
- Define $w = \xi$ (unchanged)

The Jacobian matrix \(J\) for this transformation, derived from the partial derivatives of each new variable with respect to each old variable, is given by:

$$
\begin{align}
\text{J} &= \begin{bmatrix}
\frac{\partial t_1}{\partial u} & \frac{\partial t_2}{\partial u} & \frac{\partial \xi}{\partial u} \\
\frac{\partial t_1}{\partial v} & \frac{\partial t_2}{\partial v} & \frac{\partial \xi}{\partial v} \\
\frac{\partial t_1}{\partial w} & \frac{\partial t_2}{\partial w} & \frac{\partial \xi}{\partial w}
\end{bmatrix}
= \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} \\
&= 1 \cdot (1 \cdot 1 - 0 \cdot 0) - 1 \cdot (0 \cdot 1 - 0 \cdot 0) + 0 \cdot (0 \cdot 0 - 1 \cdot 0) \\
&= 1
\end{align}
$$


Since the Jacobian determinant is $1$, the differential element $dt_2$ can be replaced by $d\Delta t$ without scaling. In the integral, if the original limits of $t_2$ are from $-\infty$ to $\infty$, the new limits of $\Delta t$ will also be from $-\infty$ to $\infty$. Now, the integral becomes:

$$
\begin{align}
\text{I} &= \int \text{F}(u, u+v, w|d_1,d_2,H) |\text{J}|\, du \, dv \, dw \\
&= \int P(t_1, \xi_1|d_1) P(t_1+\Delta t, \xi_2|d_2) \\ \nonumber
& \;\;\;\;\;\;\;\;  P(t_1, t_1+\Delta t, \xi|H) \, dt_1 \, d\Delta t \, d\xi \\
\end{align}
$$

I will factor out $t_1$ as it is independent. 

$$
\begin{align}
\text{I} &= \int P(\xi_1|t_1, d_1) P(t_1|d_1) \\ \nonumber
& \;\;\;\;\;\;\;\; P(\xi_2|t_1+\Delta t, d_2) P(t_1+\Delta t| d_2) \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta t, \xi| H) P(t_1|H)\, dt_1 \, d\Delta t \, d\xi \\
\end{align}
$$

$P(t_1, t_1+\Delta t, \xi|H) = P(t_1|H) P(\Delta t, \xi| H)$ as $t_1$ is independent of $\Delta t$ and $\xi$. Next I will consider the following assumptions:

* Since the arrival time of the gravitational waves is precise (~1ms), $P(t_1|d_1)=1$ and $P(t_2|d_2)=1$ at the arrival time of the first and second event respectively, and zero elsewhere.
* Consider $T_1$ and $T_2$ as the arrival time of the first and second event respectively. The time delay between the two events is $\Delta T = T_2 - T_1$. We labeled the two events such that the time delay is always positive, $\Delta T > 0$.

$$
\begin{align}
\text{I} &= \int P(\xi_1|T_1, d_1) P(T_1|d_1) \\ \nonumber
& \;\;\;\;\;\;\;\; P(\xi_2|T_2, d_2) P(T_2| d_2) \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, \xi|T_1, H) P(T_1|H) \, d\xi \\
&= P(T_1|H) \int P(\xi_1|T_1, d_1)\; P(\xi_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, \xi| H) \, d\xi \\
&= P(T_1|H) \int \text{KDE}(\xi_1|T_1, d_1)\; \text{KDE}(\xi_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; \text{KDE}(\Delta T, \xi| H) \, d\xi \\
\end{align}
$$


The integral is now over the joint probability distribution of the time delay and the remaining parameters. The KDE is the kernel density estimation created using the $\xi$ and re-parameterised variable $\Delta t$.

This exercise shows that if the parameter volume does not change, in this case just a shift of time parameter, the Jacobian matrix determinant is not required in the final integral. The mathematics can be exended to more lensed image condition as follows. This concept will be revisited in one of the later section.

$$
\begin{align}
\text{I} &= P(T_1|H) \int P(\xi_1|T_1, d_1)\; P(\xi_2|T_2, d_2)\; P(\xi_3|T_3, d_3) P(\xi_4|T_4, d_4)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T_{12},\Delta T_{13},\Delta T_{14}, \xi| H) \, d\xi \\
\end{align}
$$

where $\Delta T_{ij} = T_i - T_j$ and $\xi_3, \xi_4 \in \xi$. The subscript $i/j$ represents the $i^{th}/j^{th}$ event or image. Note that $\Delta T_{12},\Delta T_{13},\Delta T_{14}, \xi$ are correlated for the lensed images.

### Part 2: Distance

What if the transformation is not just a shift in the variable but a scaling? For example, consider the transformation of the luminosity distance, $d_{L,2} \rightarrow \Delta \mu .d_{L,1}$, where $\Delta \mu=d_{L,2}/d_{L,1}$. If they are lensed images, then $\Delta d_L=\sqrt{|\mu_2/\mu_1|}$. Let's consider similar integral to that of Eqn(19).

$$
\begin{align}
\text{I} &= \int P(d_{L,1},\rho_1|T_1, d_1)\; P(d_{L,2},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, d_{L,1}, d_{L,2},\rho| H) \, d\rho \,dd_{L,1}\, dd_{L,2} 
\end{align}
$$

$\rho_1, \rho_2, \rho$ are the remaining parameters in the respective probability distribution and $\rho_1, \rho_2\in \rho$. 

Following similar steps as in the time transformation, we calculated the Jacobian matrix determinant as $|J|=|d_{L,1}|=d_{L,1}$ 

$$
\begin{align}
\text{I} &= \int P(d_{L,1},\rho_1|T_1, d_1)\; P(\Delta \mu .d_{L,1},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, d_{L,1}, \Delta \mu .d_{L,1},\rho| H) d_{L,1}\, d\rho \,dd_{L,1}\, d\Delta \mu \\
&= \int \text{KDE}(d_{L,1},\rho_1|T_1, d_1)\; \text{KDE}(\Delta \mu .d_{L,1},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; \text{KDE}(\Delta T, d_{L,1}, \Delta \mu .d_{L,1},\rho| H) \, d_{L,1}\, d\rho \,dd_{L,1}\, d\Delta \mu \\
\end{align}
$$

### Part 3: KDE scaling

Let's continue with the previous integral form. In the integral, we can either rescale the probability distribution or the differential element, but not both. This means the Jacobian determinant cannot come twice. 

It is sometimes more convenient to rescale the variables first and make a KDE with the rescaled variables.

$$
\begin{align}
\text{KDE}&(d_{L,1},\rho_1|T_1, d_1) \times \text{KDE}(\Delta \mu,\rho_2|T_2, d_2) \times \\ \nonumber 
& \text{KDE}(\Delta T, d_{L,1}, \Delta \mu,\rho| H) = \text{KDE}(d_{L,1},\rho_1|T_1, d_1) \times\\ \nonumber 
& \;\;\;\;\;\;\;\;  \text{KDE}(\Delta \mu,\rho_2|T_2, d_2) \times \text{KDE}(\Delta T, d_{L,1}, \Delta \mu,\rho| H) \times d_{L,1}
\end{align}
$$

Thus, Eqn(23) becomes,

$$
\begin{align}
\text{I} &= \int \text{KDE}(d_{L,1},\rho_1|T_1, d_1)\; \text{KDE}(\Delta \mu,\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; \text{KDE}(\Delta T, d_{L,1}, \Delta \mu,\rho| H) \, d_{L,1}\, d\rho \,dd_{L,1}\, d\Delta \mu \\
\end{align}
$$

This integral form let's you use the magnification ratio as a parameter in the probability distribution, as similarly done in $\mathcal{M}_{gal}$ calculation, [More & More](https://arxiv.org/abs/2111.03091). 

**Note:** If we want a proper bayes-factor, we need to consider the scaling factor $\, d_{L,1}$ in the integral Eqn(25). I will avoid doing this, as it will make the integral more complex.

<!-- **Part 4: Functional rescaling of parameters**

Can we rescale the parameters in the probability distribution with certain functions? For example, consider the transformation $\Delta d_{L,1} \rightarrow \text{log}_{10}  d_{L,1}$. Let's start with the integral,

$$
\begin{align}
\text{I} &= \int P( d_{L,1}, \rho_1|T_1, d_1)\; P( d_{L,2}, \rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T,  d_{L,1},  d_{L,2}, \rho| H) \, d\rho \,d d_{L,1}\, d d_{L,2}
\end{align}
$$


$$
\begin{align}
\text{I} &= \int P(d_{L,1},\rho_1|T_1, d_1)\; P(d_{L,2},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, d_{L,1}, d_{L,2},\rho| H) \, d\rho \,dd_{L,1}\, dd_{L,2} \\
&= \int P(10^{\text{log}_{10}  d_{L,1}}, \rho_1|T_1, d_1)\; P(d_{L,2},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; P(\Delta T, 10^{\text{log}_{10}  d_{L,1}}, d_{L,2},\rho| H) 10^{\text{log}_{10}  d_{L,1}} \text{ln}(10) \, d\rho \,d\text{log}_{10}  d_{L,1}\, dd_{L,2} \\
&= \int \text{KDE}(\text{log}_{10}   d_{L,1},\rho_1|T_1, d_1)\; \text{KDE}(d_{L,2},\rho_2|T_2, d_2)  \\ \nonumber
& \;\;\;\;\;\;\;\; \text{KDE}(\Delta T, \text{log}_{10}  d_{L,1}, d_{L,2},\rho| H) 10^{\text{log}_{10}  d_{L,1}} \text{ln}(10) \, d\rho \,d\text{log}_{10}  d_{L,1}\, dd_{L,2}
\end{align}
$$

Let's not complicate our life and stick to the following parameter space. -->

**Part 5: Generalization**

Despite what was mentioned in the previous part, re-parameterization is still required as the PDFs needs to transform in a way that multidimensional ellipsiodal gaussian KDE can fit the data in the parameter space. This is related to the sklearn's BayesianGaussianMixture model fitting in the DPGMM model training.

But in the end the parameters are sample in the original parameter space. So, Jacobian matrix determinant is not required in the final integral.

Let's consider the following integral from Eqn (20),

$$
\begin{align}
\text{I} &=  P(T_1|H) \int \text{KDE}(\xi_1|T_1, d_1)\; \text{KDE}(\xi_2|T_2, d_2)   \text{KDE}(\Delta T, \xi| H) \, d\xi \\ \nonumber
&=  P(T_1|H) \int \text{KDE}'(R_1(\xi_1)|T_1, d_1)\; \text{KDE}(R_2(\xi_2)|T_2, d_2)   \text{KDE}'(R_3(\Delta T,\xi)| H) \, d\xi \\
\end{align}
$$

The prime in $\text{KDE}'$ represents the re-parameterized KDE. These KDEs are created using the re-parameterized variables and $R_{1,2,3}$ represents the re-parameterization or mapping functions. In the following sections, such kind of intergral will be solved numerically using monte-carlo sampling.

## Renormalization

Renormalization ensures that Kernel Density Estimation (KDE) remains a valid probability distribution within the parameter space. This necessity arises because multidimensional KDE does not always perfectly conform to the boundaries of the parameter space, resulting in a distribution that approximates the data smoothly.

### Steps for Normalizing KDE:

1. **Data Distribution**: Begin with a distribution $P(\xi|H)$ given a hypothesis $H$.

2. **Reparameterization**: Transform $\xi$ to $\xi'$ using $R(\xi) = \xi'$.

3. **Normalized KDE**: KDE created in the reparameterized space should already been normalized to 1, within some bounds [$\xi'_\text{min}$, $\xi'_\text{max}$].

   $$
   \int_{\xi'_\text{min}}^{\xi'_\text{max}} \text{KDE}'(\xi'| H) \, d\xi' = 1
   $$

4. **Constraining Parameter Space**: Set the limits $[\xi_\text{min}, \xi_\text{max}]$ in the original parameter space and convert them to the reparameterized space $[\xi''_\text{min}, \xi''_\text{max}]$. Now, compute the renormalization factor $N_{\xi'}$ as follows:

   $$
   N_{\xi'} = \int_{\xi''_\text{min}}^{\xi''_\text{max}} P(\xi'| H) \, d\xi' = \left< \frac{\text{KDE}'(\xi'| H)}{P_o(\xi')} \right>_{\xi' \in P_o(\xi')}
   $$

   where $P_o(\xi')$ is the uniform distribution over the parameter space.

5. **Final KDE Formulation**: Adjust the KDE to account for the normalization:
   $$
   \text{KDE}''(\xi'| H) = \frac{\text{KDE}'(\xi'| H)}{N_{\xi'}}
   $$

6. **Application**: Incorporate the renormalized KDE into the integral for calculations:
   $$
   \text{I} =  \frac{P(T_1|H)}{N_{\xi'_1} N_{\xi'_2} N_{\xi'}} \int \text{KDE}'(R_1(\xi_1)|T_1, d_1) \; \text{KDE}'(R_2(\xi_2)|T_2, d_2) \; \text{KDE}'(R_3(\Delta T, \xi)| H) \, d\xi
   $$

This process ensures that KDE remains accurate and representative of the parameter space, properly normalized even after transformations.

## Marginal likelihood for Lensed hypothesis

Let's discuss all the probability distributions in the integral of the Bayes factor calculation. 

$$
\begin{align}
\mathcal{B}_U^L =  \frac{ \int d\theta P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_L) }
{\int d\theta P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_U)} \\
\end{align}
$$

$\xi_1\in \{m_{1,1}, m_{2,1}, \alpha_1, \delta_1, \iota_1, d_{L,1}\}$, $\xi_2\in \{m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2, d_{L,2}\}$ and $\xi\in  \{m_{1,1}, m_{2,1}, \alpha_1, \delta_1, \iota_1, d_{L,1}, m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2, d_{L,2}\}$.

### Astrophysical prior of lensed events

$$P_{astro}(m_{1,1}, m_{2,1}, \alpha_1, \delta_1, \iota_1, d_{L,1}, m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2, d_{L,2}, \Delta T|H_L)$$

$P_{astro}(\xi,\Delta T|H_L)$: astrophysical prior on the parameters with lensed hypothesis considered. This is for the detectable events. So, the selection effect is already taken into account. The distribution preserves the correlation between the parameters. But it doesn't take into account the correlation between the coalescence time and sky location. 

$$
\begin{align}
P_{astro}(\theta|H_L) &= P_{astro}(m_{1,1}, m_{2,1}, d_{L,1}, \iota_1, \alpha_1, \delta_1, \\ \nonumber
& \;\;\;\;\;\;\;\; m_{1,2}, m_{2,2}, d_{L,2}, \iota_2, \alpha_2, \delta_2, \Delta T |H_L) \\
&= P_{astro}(m_{1,2}, m_{2,2}, \iota_2, \alpha_2, \delta_2| \\ \nonumber
& \;\;\;\;\;\;\;\; m_{1,1}, m_{2,1}, \iota_1, \alpha_1, \delta_1, d_{L,1}, d_{L,2}, \Delta T, H_L) \times \\ \nonumber
& \;\;\;\;\;\;\;\; P_{astro}(m_{1,1}, m_{2,1}, \iota_1, \alpha_1, \delta_1, d_{L,1}, d_{L,2}, \Delta T | H_L) \\
&= 1 \times P_{astro}(m_{1,1}, m_{2,1}, \iota_1, \alpha_1, \delta_1, d_{L,1}, d_{L,2}, \Delta T | H_L)
\end{align}
$$

Going from Eqn(29) to Eqn(30), I have used the lensed hypothesis to collapse the parameters of the second image, i.e. the mass, inclination angle, and sky location of the second image are the same as the first image. As a consequence, $P_{astro}(m_{1,2}, m_{2,2}, \iota_2, \alpha_2, \delta_2| m_{1,1}, m_{2,1}, \iota_1, \alpha_1, \delta_1, d_{L,1}, d_{L,2}, \Delta T, H_L)$ is a delta function centre at the parameters of image 1. The luminosity and the time delay are the only parameters that are different between the two images. I have ignored the phase difference between the two images for my study. Below, I am separating out the un-correlated parameters and correlated parameters.

$$
\begin{align}
P_{astro}(\theta|H_L) 
&= P_{astro}(m_{1,1}, m_{2,1}, \iota_1, \Delta \mu, \Delta T | H_L)
P_{astro}(\alpha_1, \delta_1 | H_L) 
\end{align}
$$

### Posterior distributions

$$P(m_{1,1}, m_{2,1}, \alpha_1, \delta_1, \iota_1, d_{L,1}|T_1, d_1) \text{ and } P(m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2, d_{L,2}|T_2, d_2)$$

#### First event

$P(\xi_1|T_1,d_1)$: Posterior distribution of the parameters given the data from the first GW event and the coalescence time of the first GW event.

$$
\begin{align}
P(\theta|d_1) &= P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1) P(\alpha_1, \delta_1 | T_1, d_1) \\
\end{align}
$$

#### Second event

$P(\xi_2|T_2,d_2)$: Posterior distribution of the parameters given the data from the second GW event and the coalescence time of the second GW event.

$$
\begin{align}
P(\theta|d_2) &= P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2) P(\alpha_2, \delta_2 | T_2, d_2) \\
\end{align}
$$

### Astrophysical prior of unlensed events

$$P_{astro}(m_{1,1}, m_{2,1}, \alpha_1, \delta_1, \iota_1, d_{L,1}, m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2, d_{L,2}, \Delta T|H_U)$$

$P_{astro}(\xi,\Delta T|H_U)$: astrophysical prior on the parameters with unlensed hypothesis considered. This is for the detectable events. So, the selection effect is already taken into account. The distribution preserves the correlation between the parameters. But it doesn't take into account the correlation between the coalescence time and sky location.

$$
\begin{align}
P_{astro}(\theta|H_U) &= P_{astro}(m_{1,1}, m_{2,1}, d_{L,1}, \iota_1, \alpha_1, \delta_1, m_{1,2}, m_{2,2}, d_{L,2}, \iota_2, \alpha_2, \delta_2, \Delta T |H_U) \\
&= P_{astro}(m_{1,1}, m_{2,1}, d_{L,1}, \iota_1|H_U) \times  P_{astro}(m_{1,2}, m_{2,2}, d_{L,2}, \iota_2, \Delta T |H_U) \times \\
&\;\;\;\;\;\;\;\; P_{astro}(\alpha_1, \delta_1 |H_U) \times P_{astro}(\alpha_2, \delta_2 |H_U) \\
\end{align}
$$

In the last step, I have assumed that the right ascension and declination distributions are independent of the other parameters.


## Numerical method

### Introduction Monte-Carlo Integration

If $P(x)$ is a normalized probability distribution, then the expectation value of a function $f(x)$ is given by the integral:

$$
\begin{align}
I = \int f(x) P(x) dx = \langle f(x) \rangle _{x\in P(x)}
\end{align}
$$

### Sampling in the parameter space

The integration in numerator and denominator of the Bayes factor is done using Monte-Carlo integration. The integral is approximated as the sum of the integrand evaluated at random samples of the parameter space. The samples need to be drawn from normalized PDFs. The question is which PDFs to consider for the sampling. 

* The following PDFs are considered for the sampling in the denominator of the Bayes factor calculation:
    * $P(\xi_1|T_1,d_1) \rightarrow \text{KDE}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1)\times \text{KDE}( \alpha_1, \delta_1|T_1, d_1)$  
    * $P(\xi_2|T_2,d_2) \rightarrow \text{KDE}(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2) \times \text{KDE}( \alpha_2, \delta_2|T_2, d_2)$

* For the numerator of the Bayes factor calculation, I will construct a joint PDF as follows:
    * $P(\xi_1|T_1,d_1)\cup P(\xi_2|T_2,d_2) \rightarrow \text{KDE}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2) \times \text{KDE}( \alpha, \delta|T_1, d_1, T_2, d_2)$
    * This is with the consideration of lensing hypothesis. Some of the parameters ($m_{1,2}, m_{2,2}, \alpha_2, \delta_2, \iota_2$) of the second event are collapsed to that of the first event. The common parameters are $m_{1}$, $m_{2}$, $\alpha$, $\delta$, $\iota$.
    * Prior re-weighting is done to use this joint PDF in the numerator.


### Bayes factor, analytical form

$$
\begin{align}
\mathcal{B}_U^L &= \frac{ \int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_L) }
{\int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_U)} \\
\end{align}
$$

#### Numerator

$$
\begin{align}
I_1 &= \int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_L) \\
&= \int dm_{1}\, dm_{2}\, d\iota\, dd_{L,1}\, dd_{L,2} \, d\alpha\, d\delta \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,1}|T_1, d_1) P(\alpha, \delta | T_1, d_1) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,2}|T_2, d_2) P(\alpha, \delta | T_2, d_2) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, \Delta T|H_L) P_{astro}(\alpha, \delta | H_L) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2) P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2) \div \\ \nonumber
&\;\;\;\;\;\;\;\; \{P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2) P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2)\} \\
\end{align}
$$

#### Denominator

$$
\begin{align}
I_2 &= \int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P_{astro}(\xi,\Delta T|H_U) \\
&= \int dm_{1,1}\, dm_{2,1}\, d\iota_1\, dd_{L,1}\, d\alpha_1\, d\delta_1 \times \\ \nonumber
&\;\;\;\;\;\;\;\; dm_{1,2}\, dm_{2,2}\, d\iota_2\, dd_{L,2}\, d\alpha_2\, d\delta_2 \times \\ \nonumber
&\;\;\;\;\;\;\;\;P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1) P(\alpha_1, \delta_1 | T_1, d_1) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2) P(\alpha_2, \delta_2 | T_2, d_2) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|H_U) P(\alpha_1, \delta_1 | H_U) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,2}, m_{2,2},  \iota_2, d_{L,2}, \Delta T|H_U) P(\alpha_2, \delta_2 | H_U)
\end{align}
$$

Note that $P_{astro}(m_{1,2}, m_{2,2},  \iota_2, d_{L,2}, \Delta T|H_U)$ can also be written as $P_{astro}(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|H_U) P_{astro}(\Delta T|H_U)$, as the time delay is independent of the other parameters for unlensed hypothesis. 

### Bayes factor, numerical integration

Numerical integration form of the Bayes factor is,

$$
\begin{align}
\mathcal{B}_U^L = \Bigg[&\\ \nonumber
& \bigg< \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,1}|T_1, d_1) P(\alpha, \delta | T_1, d_1) \times \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,2}|T_2, d_2) P(\alpha, \delta | T_2, d_2) \times \\ \nonumber
&\; P_{astro}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, \Delta T|H_L) P_{astro}(\alpha, \delta | H_L) \div \\ \nonumber
&\; \{P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2) P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2)\} \\ \nonumber
&\bigg>_{ m_1, m_2, \iota, d_{L,1}, d_{L,2} \in P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2),\; \alpha, \delta \in P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2)} \\ \nonumber
& \Bigg] \div \Bigg[P_{astro}(\Delta T|H_U) \times \\ \nonumber
& \bigg<\\ \nonumber
&\; P_{astro}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|H_U) P(\alpha_1, \delta_1 | H_U) \times \\ \nonumber
&\; P_{astro}(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|H_U) P(\alpha_2, \delta_2 | H_U) \\ \nonumber
&\bigg>_{ m_{1,1}, m_{2,1}, \iota_1, d_{L,1}\in P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1),\; \alpha_1, \delta_1 \in P(\alpha_1, \delta_1 | T_1, d_1),}\\ \nonumber
&_{\; m_{1,2}, m_{2,2}, \iota_2, d_{L,2}\in P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2),\; \alpha_1, \delta_1 \in P(\alpha_2, \delta_2 | T_2, d_2)} \\ \nonumber
\Bigg]& 
\end{align}
$$

## Extending posterior overlap to four images

### Bayes factor, analytical form

Starting from Eqn(21), we can write the integral for the four images as,

#### Numerator

$$
\begin{align}
I_1 &= \int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P(\xi_3|T_3,d_3) P(\xi_4|T_4,d_4) \times \\ \nonumber 
&\;\;\;\;\;\;\;\; P_{astro}(\xi,\Delta T_{12}, \Delta T_{13}, \Delta T_{14}|H_L) \\

&= \int dm_{1}\, dm_{2}\, d\iota\, dd_{L,1}\, dd_{L,2} \, dd_{L,3}\, dd_{L,4} \, d\alpha\, d\delta \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,1}|T_1, d_1) P(\alpha, \delta | T_1, d_1) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,2}|T_2, d_2) P(\alpha, \delta | T_2, d_2) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,3}|T_3, d_3) P(\alpha, \delta | T_3, d_3) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_1, m_2, \iota, d_{L,4}|T_4, d_4) P(\alpha, \delta | T_4, d_4) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4},\Delta T_{12}, \Delta T_{13}, \Delta T_{14}|H_L) P_{astro}(\alpha, \delta | H_L) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{comb}(m_1, m_2, \alpha, \delta, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4}|T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4) \div \\ \nonumber
&\;\;\;\;\;\;\;\; \{P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4}|T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4) \times\\ \nonumber 
&\;\;\;\;\;\;\;\; P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4)\} \\
\end{align}
$$

#### Denominator

$$
\begin{align}
I_2 &= \int d\xi P(\xi_1|T_1,d_1) P(\xi_2|T_2,d_2) P(\xi_3|T_3,d_3) P(\xi_4|T_4,d_4) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(\xi,\Delta T_{12}, \Delta T_{13}, \Delta T_{14}|H_U) \\
&= \int dm_{1,1}\, dm_{2,1}\, d\iota_1\, dd_{L,1}\, d\alpha_1\, d\delta_1 \times \\ \nonumber
&\;\;\;\;\;\;\;\; dm_{1,2}\, dm_{2,2}\, d\iota_2\, dd_{L,2}\, d\alpha_2\, d\delta_2 \times \\ \nonumber
&\;\;\;\;\;\;\;\; dm_{1,3}\, dm_{2,3}\, d\iota_3\, dd_{L,3}\, d\alpha_3\, d\delta_3 \times \\ \nonumber
&\;\;\;\;\;\;\;\; dm_{1,4}\, dm_{2,4}\, d\iota_4\, dd_{L,4}\, d\alpha_4\, d\delta_4 \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1) P(\alpha_1, \delta_1 | T_1, d_1) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2) P(\alpha_2, \delta_2 | T_2, d_2) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_{1,3}, m_{2,3}, \iota_3, d_{L,3}|T_3, d_3) P(\alpha_3, \delta_3 | T_3, d_3) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P(m_{1,4}, m_{2,4}, \iota_4, d_{L,4}|T_4, d_4) P(\alpha_4, \delta_4 | T_4, d_4) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|H_U) P(\alpha_1, \delta_1 | H_U) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,2}, m_{2,2},  \iota_2, d_{L,2}, \Delta T_{12}|H_U) P(\alpha_2, \delta_2 | H_U) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,3}, m_{2,3}, \iota_3, d_{L,3}, \Delta T_{13}|H_U) P(\alpha_3, \delta_3 | H_U) \times \\ \nonumber
&\;\;\;\;\;\;\;\; P_{astro}(m_{1,4}, m_{2,4}, \iota_4, d_{L,4}, \Delta T_{14}|H_U) P(\alpha_4, \delta_4 | H_U) \\
\end{align}
$$

Bayes factor is calculated using the following formula,

$$
\begin{align}
\mathcal{B}_U^L = \Bigg[&\\ \nonumber
& \bigg< \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,1}|T_1, d_1) P(\alpha, \delta | T_1, d_1) \times \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,2}|T_2, d_2) P(\alpha, \delta | T_2, d_2) \times \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,3}|T_3, d_3) P(\alpha, \delta | T_3, d_3) \times \\ \nonumber
&\; P(m_1, m_2, \iota, d_{L,4}|T_4, d_4) P(\alpha, \delta | T_4, d_4) \times \\ \nonumber
&\; P_{astro}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4},\Delta T_{12}, \Delta T_{13}, \Delta T_{14}|H_L) P_{astro}(\alpha, \delta | H_L) \div \\ \nonumber
&\; \{P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4}|T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4) \times \\ \nonumber
&\; \; P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4)\} \\ \nonumber
&\bigg>_{ m_1, m_2, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4} \in P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}, d_{L,3}, d_{L,4}|T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4),} \\ \nonumber
&_{\;\;\;\, \alpha, \delta \in P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2, T_3, d_3, T_4, d_4)} \\ \nonumber
& \Bigg] \div \Bigg[ P(\Delta T_{12}|H_U) P(\Delta T_{13}|H_U) P(\Delta T_{14}|H_U)  \times \\ \nonumber
& \bigg<\\ \nonumber
&\; P_{astro}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|H_U) P(\alpha_1, \delta_1 | H_U) \times \\ \nonumber
&\; P_{astro}(m_{1,2}, m_{2,2},  \iota_2, d_{L,2}|H_U) P(\alpha_2, \delta_2 | H_U) \times \\ \nonumber
&\; P_{astro}(m_{1,3}, m_{2,3}, \iota_3, d_{L,3}|H_U) P(\alpha_3, \delta_3 | H_U) \times \\ \nonumber
&\; P_{astro}(m_{1,4}, m_{2,4}, \iota_4, d_{L,4}|H_U) P(\alpha_4, \delta_4 | H_U) \\ \nonumber
&\bigg>_{ m_{1,1}, m_{2,1}, \iota_1, d_{L,1}\in P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1),\; \alpha_1, \delta_1 \in P(\alpha_1, \delta_1 | T_1, d_1),}\\ \nonumber
&_{\;\;\;\, m_{1,2}, m_{2,2}, \iota_2, d_{L,2}\in P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2),\; \alpha_1, \delta_1 \in P(\alpha_2, \delta_2 | T_2, d_2),}\\ \nonumber
&_{\;\;\;\, m_{1,3}, m_{2,3}, \iota_3, d_{L,3}\in P(m_{1,3}, m_{2,3}, \iota_3, d_{L,3}|T_3, d_3),\; \alpha_1, \delta_1 \in P(\alpha_3, \delta_3 | T_3, d_3),}\\ \nonumber
&_{\;\;\;\, m_{1,4}, m_{2,4}, \iota_4, d_{L,4}\in P(m_{1,4}, m_{2,4}, \iota_4, d_{L,4}|T_4, d_4),\; \alpha_1, \delta_1 \in P(\alpha_4, \delta_4 | T_4, d_4)} \\ \nonumber
\Bigg]&
\end{align}
$$






<!-- Prior probability distributions from where the samples are drawn are listed below:

* $P_{comb}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2)$ or $\text{KDE}(m_1, m_2, \iota, d_{L,1}, d_{L,2}|T_1, d_1, T_2, d_2)$: Combined prior distribution of the parameters for the two events.
* $P_{comb}(\alpha, \delta | T_1, d_1, T_2, d_2)$ or $\text{KDE}(\alpha, \delta | T_1, d_1, T_2, d_2)$: Combined prior distribution of the sky location for the two events.
* $P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1)$ or $\text{KDE}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1)$: Posterior distribution of the parameters given the data from the first GW event and the coalescence time of the first GW event.
* $P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2)$ or $\text{KDE}(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2)$: Posterior distribution of the parameters given the data from the second GW event and the coalescence time of the second GW event.
* $P(\alpha_1, \delta_1 | T_1, d_1)$ or $\text{KDE}(\alpha_1, \delta_1 | T_1, d_1)$: Posterior distribution of the sky location given the data from the first GW event and the coalescence time of the first GW event.
* $P(\alpha_2, \delta_2 | T_2, d_2)$ or $\text{KDE}(\alpha_2, \delta_2 | T_2, d_2)$: Posterior distribution of the sky location given the data from the second GW event and the coalescence time of the second GW event.

Probability distributions from where the PDF values are calculated are listed below:

* $P(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1)$ or $\text{KDE}(m_{1,1}, m_{2,1}, \iota_1, d_{L,1}|T_1, d_1)$: Posterior distribution of the parameters given the data from the first GW event and the coalescence time of the first GW event.
* $P(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2)$ or $\text{KDE}(m_{1,2}, m_{2,2}, \iota_2, d_{L,2}|T_2, d_2)$: Posterior distribution of the parameters given the data from the second GW event and the coalescence time of the second GW event.
* $P(\alpha_1, \delta_1 | T_1, d_1)$ or $\text{KDE}(\alpha_1, \delta_1 | T_1, d_1)$: Posterior distribution of the sky location given the data from the first GW event and the coalescence time of the first GW event.
* $P(\alpha_2, \delta_2 | T_2, d_2)$ or $\text{KDE}(\alpha_2, \delta_2 | T_2, d_2)$: Posterior distribution of the sky location given the data from the second GW event and the coalescence time of the second GW event.
* $P_{astro}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, \Delta T|H_L)$ or $\text{KDE}(m_{1}, m_{2}, \iota, d_{L,1}, d_{L,2}, \Delta T|H_L)$: Astrophysical prior on the parameters with lensed hypothesis considered.
* $P_{astro}(\alpha, \delta | H_L)$ or $\text{KDE}(\alpha, \delta | H_L)$: Astrophysical prior on the sky location with lensed hypothesis considered. -->