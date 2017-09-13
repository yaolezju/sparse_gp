% GAUSSIAN PROCESS REGRESSION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now, we compute the predictions for the ASA dataset using a Gaussian 
% process regressin with a squared exponential covariance function with a 
% noise term.
% We need specify the mean, covariance and likelihood functions.

% Split the data 'X' we got from 'PreprocessingASA.m' into train and test 
% set. The ratio used in the paper was 70%  train and 30% test.
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
covn = {'covSum', {'covSEard','covNoise'}}; feval(covn{:});
likfunc = @likGauss;              % Gaussian likelihood   


% Initialize the hyperparameter struct
% A common mistake in these situations is when we initialise parameters
% without a clear idea. A common task is to generate the hyperparameters 
% by optimising the log marginal likelihood. 
% So we set the struct of the hyperparameters, the gp function, the number 
% of function evaluation (100), and the mean, cov, and likeli functions.
hyp = struct('mean', [], 'cov', [0 0 0 0 0 0 0 0 0 log(0.1)], 'lik', -1);


% II.  GAUSSIAN PROCESS REGRESSION - FITC and VFE

  
% Ia.2 - Inducing points (Rasmussen's book) and regression
% 50 -- 16390 A
% 100 -- 8190 B
% 250 -- 3270 C
% 500 -- 1637 D
% 1000 -- 819.5 E
nu = fix(n/16390); 
% Be careful bellow, there are some issues when we use {cov} in composite
% functions!!!
covfuncF = {'apxSparse', covn, u}; % Set covariance function for inducing

% Choose the inference method:
%inf = @(varargin) infGaussLik(varargin{:}, struct('s', 1.0)); % Gauss with FITC
inf = @(varargin) infGaussLik(varargin{:}, struct('s', 0.0)); % Gauss VFE   

% Create the tables to hold the values from the GPR tests.
% For the VFE tests we named our tables as:
%       mu = predicted means
%            mu50, mu100, mu250, mu500
%       s = predicted variances
%            s50, s100, s250, s500
%       table = contains MSE, negative log likel, run time
%            table50, table100, table250, table500
% For the FITC we named the tables:
%       muF = predicted means
%            muF50, muF100, muF250, muF500
%       similar to VFE for the rest (s, table).

mu50 = zeros(length(xz),10);
s50 = zeros(length(xz),10);
table50 = zeros(10,3); % Contains MSE, nlZ, time

for i=1:10
    iu = randperm(n); iu = iu(1:nu); u = x(iu,:);
    covfuncF = {'apxSparse', covn, u};
    hyp.xu = u;
    tic;
    hyp3 = minimize(hyp, @gp, -100, inf, meanfunc, covfuncF, likfunc, x, y); 
    [mu s2] = gp(hyp3, inf, meanfunc, covfuncF, likfunc, x, y, xz); % dense prediction
    table50(i,3) = toc;
    %mu50(:,i) = mu;
    %s50(:,i) = s2;
    %[nlZ,dnlZ] = gp(hyp2, inf, meanfunc, covfuncF, likfunc, x, y); % marginal likelihood and derivatives
    %table50(i,2) = nlZ;
    MSE = immse(mu,yz); % mean square error
    table50(i,1) = MSE;
    disp(i);  
end

mean(table50);
std(table50);










