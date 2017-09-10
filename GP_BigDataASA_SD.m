% GAUSSIAN PROCESS REGRESSION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now, we compute the predictions for the ASA dataset using a Gaussian 
% process regressin with a squared exponential covariance function with a 
% noise term.
% We need specify the mean, covariance and likelihood functions.

% Split the data 'X' we got from 'PreprocessingASA.m' into train and test 
% set. The ratio used in the paper was 70%  train and 30% test.

% First let's create a small sample for the optimization of 10000
% observations. We will use this for the SD (Subset of Data) Gaussian 
% process regression (GPR) method.
[Xtrain,~,Xtest] = dividerand(transpose(X),0.991,0,0.009);
Xtrain = transpose(Xtrain);
Xtest = transpose(Xtest);
% Name it x_obs
x_obs = Xtest;
y_obs = x_obs(:,9);
x_obs = x_obs(:,1:8);
% Again do the split but this time for training and test sets
[Xtrain,~,Xtest] = dividerand(transpose(X),0.7,0,0.3);
Xtrain = transpose(Xtrain);
Xtest = transpose(Xtest);

% Further split the train and test between responce and explanaroty
x = Xtrain(:,1:8); 
y = Xtrain(:,9);
xz = Xtest(:,1:8); 
yz = Xtest(:,9);
n = size(x,1);
testn = size(xz,1);


% I.  COVARIANCE MATRIX APROXIMATION (Inducing points)
% Ia. GP with exact Gaussian inference or VB
%     SE covariance function or SE ARD
%     FITC approximation or 

% Ia.1 - Parameters
meanfunc = [];          % empty: don't use a mean function

% composite squared exponential covariance function with ARD and Noise
covn = {'covSum', {'covSEard','covNoise'}}; %feval(covn{:});
likfunc = @likGauss;              % Gaussian likelihood   

% Initialize the hyperparameter struct
% A common mistake in these situations is when we initialise parameters
% without a clear idea. A common task is to generate the hyperparameters 
% by optimising the log marginal likelihood. 
% So we set the struct of the hyperparameters, the gp function, the number 
% of function evaluation (100), and the mean, cov, and likeli functions.
hyp = struct('mean', [], 'cov', [0 0 0 0 0 0 0 0 0 log(0.1)], 'lik', -1);

% Optimise hyp using exact inference
infg = @infGaussLik;

% For the optimization part we followed Rasmussen and Williams (2006)
% experiment that optimised the covariance hyperparameters only once and on
% a small dataset (x_obs, y_obs).
hyp2 = minimize(hyp, @gp, -100, infg, meanfunc, covn, likfunc, x_obs, y_obs); 


% II.  GAUSSIAN PROCESS REGRESSION - SUBSET OF DATA
% This is my idea to split the data into mini-batches
% Split the dataset into smaller batches with 70% train and 30% test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manually change the name of the mu, s2, and Table for each mini-set
% Example:
%           mu_1 = subsets of n=250; test size n=75; iterations = 4683
%           mu_2 = subsets of n=500; test size n=150; iterations = 2341
%           mu_3 = subsets of n=1000; test size n=300; iterations = 1170
%           mu_4 = subsets of n=1700; test size n=510; iterations = 688
%
% Also, we manually change the mu, s2, and Table depending on the inference 
% method:
%           mu, s2, Table = exact inference
%           Lmu, Ls2, LTable = Laplace inference
%           VBmu, VBs2, VBTable = Variational Bayes inference

% Exact Inference
mu_1 = zeros(75,4683); % Test size/Iterations
s2_1 = zeros(75,4683);
Table1 = zeros(4683,2);
nn = 75;
prev_batch = 0;

inf = @infGaussLik;      %Exact inference 

for k=1:4683
    tic;
    new_batch = k*250; % Size of the mini-batch (train+test)
    X_batch = X(1+prev_batch:new_batch,:);
    % Random train/test
    [train,~,test] = dividerand(transpose(X_batch),0.7,0,0.3);
    train = transpose(train);
    test = transpose(test);
    x_t = train(:,1:8); % Train batch 
    y_t = train(:,9);
    x_z = test(:,1:8); % Test batch
    y_z = test(:,9);
    % Perform regression using the hyper we optimised in STEP I
    [m_ s_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t, x_z);
    [nlZ_,dnlZ_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t);
    Table1(k,1) = immse(m_,y_z); % The mean square error
    Table1(k,2) = nlZ_; % Save the negative log likelihood
    mu_1(:,k) = m_; % Save the predicted mean
    s2_1(:,k) = s_; % Save the predicted variance
    prev_batch = new_batch;
    p4 = toc;
    disp(k);
end

SumTable1 = [mean(Table1(:,1)) std(Table1(:,1)); mean(Table2(:,1)) ...
    std(Table2(:,1)); mean(Table3(:,1)) std(Table3(:,1)); mean(Table4(:,1)) ...
    std(Table4(:,1))];

% Laplace inference
Lmu_4 = zeros(510,688); 
Ls2_4 = zeros(510,688);
LTable4 = zeros(688,2);
nn = 510;
prev_batch = 0;

inf = @infLaplace;       %Laplace's Approximation

for k=1:688
    tic;
    new_batch = k*1700;
    X_batch = X(1+prev_batch:new_batch,:);
    [train,~,test] = dividerand(transpose(X_batch),0.7,0,0.3);
    train = transpose(train);
    test = transpose(test);
    x_t = train(:,1:8); % Train batch 
    y_t = train(:,9);
    x_z = test(:,1:8); % Test batch
    y_z = test(:,9);
    % Perform regression using the hyper we optimised in STEP I
    [m_ s_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t, x_z);
    [nlZ_,dnlZ_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t);
    LTable4(k,1) = immse(m_,y_z); % The mean square error
    LTable4(k,2) = nlZ_; % Save the negative log likelihood
    Lmu_4(:,k) = m_; % Save the predicted mean
    Ls2_4(:,k) = s_; % Save the predicted variance
    prev_batch = new_batch;
    Lp4 = toc;
    disp(k);
end

SumLTable1 = [mean(LTable1(:,1)) std(LTable1(:,1)); mean(LTable2(:,1))...
    std(LTable2(:,1)); mean(LTable3(:,1)) std(LTable3(:,1)); ...
    mean(LTable4(:,1)) std(LTable4(:,1))];

% Variational Bayes inference
VBmu_4 = zeros(510,688);  
VBs2_4 = zeros(510,688);
VBTable4 = zeros(688,2);
nn = 510;
prev_batch = 0;

inf = @infVB; % VB inference

for k=1:688
    tic;
    new_batch = k*1700;
    X_batch = X(1+prev_batch:new_batch,:);
    [train,~,test] = dividerand(transpose(X_batch),0.7,0,0.3);
    train = transpose(train);
    test = transpose(test);
    x_t = train(:,1:8); % Train batch 
    y_t = train(:,9);
    x_z = test(:,1:8); % Test batch
    y_z = test(:,9);
    [m_ s_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t, x_z);
    [nlZ_,dnlZ_] = gp(hyp2, inf, meanfunc, covn, likfunc, x_t, y_t);
    VBTable4(k,1) = sqrt(sum(((m_-y_z).^2))/nn);
    VBTable4(k,2) = nlZ_;
    VBmu_4(:,k) = m_;
    VBs2_4(:,k) = s_;
    prev_batch = new_batch;
    VBp4 = toc;
    disp(k);
end

SumVBTable1 = [mean(VBTable1(:,1)) std(VBTable1(:,1)); mean(VBTable2(:,1))...
    std(VBTable2(:,1)); mean(VBTable3(:,1)) std(VBTable3(:,1));...
    mean(VBTable4(:,1)) std(VBTable4(:,1))];
