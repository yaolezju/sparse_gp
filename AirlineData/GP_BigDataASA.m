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
% The covariance hyperparameters should be 8 (for the dimensions) + 1 for
% the signal noise + 1 for the covNoise.
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

% Count time
mu = zeros(length(xz),10);
s = zeros(length(xz),10);
table = zeros(10,3); % Contains rMSE, nlZ, time

for i=1:10
    iu = randperm(n); iu = iu(1:nu); u = x(iu,:);
    covfuncF = {'apxSparse', covn, u};
    hyp.xu = u;
    tic;
    hyp3 = minimize(hyp, @gp, -100, inf, meanfunc, covfuncF, likfunc, x, y); 
    [mu s2] = gp(hyp3, inf, meanfunc, covfuncF, likfunc, x, y, xz); % dense prediction
    table(i,3) = toc;
    %mu(:,i) = mu;
    %s(:,i) = s2;
    %[nlZ,dnlZ] = gp(hyp2, inf, meanfunc, covfuncF, likfunc, x, y); % marginal likelihood and derivatives
    %table(i,2) = nlZ;
    rMSE = immse(mu,yz); % mean square error
    table(i,1) = rMSE;
    disp(i);  
end

mean(table);
std(table);










