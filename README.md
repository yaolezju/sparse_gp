# sparse_gp
This is the repository for my MSc thesis for the CSML at UCL with title *A Framework for Evaluating Sparse
Gaussian Processes*. 

The theme of the thesis is sparse Gaussian processes and is based on previous work of Titsias (2009), Hensman et al, (2013), and Ghahramani and Snelson (2006).

My aim is to continue working on this as I have identified aspects on SVI that I want to test and might later improve. Also, I would like to try and incorporate the SVI model on the GPML library.

**From the abstract:**

This work introduces the concept of sparse Gaussian processes, which is based on the re-parametrisation of the Gaussian process. We 
thoroughly investigate and analyse the most popular sparse Gaussian process methods, including the Fully Independent Training 
Conditionals (FITC), the Variational Free Energy (VFE), and the Stochastic Variational Inference (SVI).

We perform a series of Gaussian process regression analysis tests on two datasets, one large with over a million observations 
(air traffic dataset) and a medium one with 400,000 observations (kin40k). Our aim is to predict, with the minimum error, 
the flight delay in the first dataset and the distance between a robotic arm and its target object in the second dataset. 

We show that the Stochastic Variational Inference model outperforms all the other sparse solutions in both the average
mean squared error (MSE) and the run time, with computational cost O(m^3) where m the inducing points of the model.

**Code:**

The code includes files written in MATLAB and in Python. 
Specifically, the Fully Independent Training Conditionals (FITC) and the Variational Free Energy (VFE) models were ran on GPML Toolbox on MATLAB. Thus, for the user to run them is necessary to download the GPML files from:
http://www.gaussianprocess.org/gpml/code/matlab/doc/

For the Stochastic Variational Inference (SVI) we used the Python library GPflow:
http://gpflow.readthedocs.io/en/latest/

The parts are:
1. Folder (AirlineData):
    * *PreprocessingASA.m*: Data preprocessing in MATLAB for the creation of the covariate matrix X.
    * *GP_BigDataASA_SD.m*: Code to run the Subset of Data (SD) method.
    * *GP_BigDataASA.m*: Code to run the FITC and VFE methods.
    * *GraphsandPlots.m*: Code to generate the graphs of all three methods (SD, FITC, VFE).
    * *SVI.py*: Code for the SVI from Hensman et al, (2013) on the airline data.
2. Folder (kin40k):
    * For the kin40k method we performed the exact same steps, except for the preprocessing were we did not needed it. We will upload the code on a later time.

Steps:
* *For the Airline Data*: To run the FITC, VFE and Subset of Data (SD) that were presented on the paper initially we execute the PreprocessingASA.m. Then we run the GP_BigDataASA_SD.m to perform the SD regression computations. Following, we run the GP_BigDataASA.m for the FITC and VFE tests. Afterwords, we generate the graphs by running the GraphsandPlots.m. Finally, run the SVI.py for the SVI method.
* *For the kin40k Data*: We follow the same steps except the preprocessing, as the data have already been preprocessed. We will update the folder with specific tailored codes for kin40k in due time.

**Bibliography:**

Ghahramani, Z. and Snelson, E. (2006). Sparse Gaussian Processes using Pseudo-inputs. *Advances in Neural Information Processing Systems*, 19, pp.1257–1264.

Hensman, J., Fusi, N. and Lawrence, N. (2013). Gaussian Processes for Big Data. *Uncertainty in Artiﬁcial Intelligence*, 29.

Titsias, M. (2009). Variational Learning of Inducing Variables in Sparse Gaussian Processes. *Artificial Intelligence and Statistics* 12, 5, pp.567–574.
