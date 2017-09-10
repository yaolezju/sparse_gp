# sparse_gp
This is the repository for my MSc thesis for the CSML at UCL with title *A Framework for Evaluating Sparse
Gaussian Processes*. 

The theme of the thesis is sparse Gaussian processes and is based on previous work of Titsias (2009), Hensman et al, (2013), and Ghahramani and Snelson (2006).

**From the abstract:**

This work introduces the concept of sparse Gaussian processes, which is based on the re-parametrisation of the Gaussian process. We 
thoroughly investigate and analyse the most popular sparse Gaussian process methods, including the Fully Independent Training 
Conditionals (FITC), the Variational Free Energy (VFE), and the Stochastic Variational Inference (SVI).

We perform a series of Gaussian process regression analysis tests on two datasets, one large with over a million observations 
(air traffic dataset) and a medium one with 400,000 observations (kin40k). Our aim is to predict, with the minimum error, 
the flight delay in the first dataset and the distance between a robotic arm and its target object in the second dataset. 
We show that the Stochastic Variational Inference model outperforms all the other sparse solutions in both the average
mean squared error (MSE) and the run time, with computational cost O($m^3$) where $m$ the inducing points of the model.

**Bibliography:**

Ghahramani, Z. and Snelson, E. (2006). Sparse Gaussian Processes using Pseudo-inputs. *Advances in Neural Information Processing Systems*, 19, pp.1257–1264.

Hensman, J., Fusi, N. and Lawrence, N. (2013). Gaussian Processes for Big Data. *Uncertainty in Artiﬁcial Intelligence*, 29.

Titsias, M. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes. *Artificial Intelligence and Statistics* 12, 5, pp.567–574.
