% This is the code to run preprocessing steps for the Fully Independent 
% Training Conditionals (FITC)[1], and the Variational Free Energy (VFE)[2] 
% models. For this example we use the same dataset (flight delays) that 
% Hensman used in 'GP for big Data' (2013). We then compare the results 
% with his paper.

% [1] Ghahramani, Z. and Snelson, E. (2006). Sparse Gaussian Processes 
%     using Pseudo-inputs. Advances in Neural Information Processing 
%     Systems, 19, pp.1257–1264.
%
% [2] Titsias, M. (2009). Variational Learning of Inducing Variables in 
%     Sparse Gaussian Processes. Artificial Intelligence and Statistics 
%     12, 5, pp.567–574.
%
% The link for the dataset is: 
% http://stat-computing.org/dataexpo/2009/the-data.html
% 


%%%%%%%%%%%%%%%%%%%%%%%%% Preprocessing State %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import the CSV file with all 7 million rows. 
Airline_Data = readtable('2008.csv');
Planes = readtable('plane-data.csv');

% Calculate the years that a plane has been in service
Planes.year = str2double(Planes.year); % make it to double to be compatible
                                       % with Airline_Data Year
Planes.Properties.VariableNames{'tailnum'} = 'TailNum'; % Change the tailnum
                                                        % to TailNum to
                                                        % create a common
                                                        % key

% Join the two tables together (left join) to calculate the age of the
% plane
A = outerjoin(Airline_Data,Planes,'Type','Left','MergeKeys',true);
A.AircraftAge = A.Year - A.year;

% So now we have created a new combined table named A

% Extract the first five rows from the A table
A(5,:);

% Extract data from the table 'A' for 4  months (Jan-Apr)
Data = A(A.Month==1 | A.Month==2 | A.Month==3 | A.Month==4,:);
Data(5,:);

% Produce explanatory statistics for the data. 
% First we will check how NA values each column has
%summary(Data);
%mean = nanmean(Data{:,15},1);
%M = ismissing(Data); % Create a logical with the missing values
%nan_sum = sum(M);
%NAN = nan_sum(:, nan_sum(:,:)>0); % Find the number of missing values
%unqValue=unique(Data); % count unique values
%n=arrayfun(@(x) sum(Data==x),unqValue);

% How the paper found 30 min delay. Just delete negative and NAN values
Data.ArrDelay = str2double(Data.ArrDelay);
nanmean(Data.ArrDelay);
Data = Data(~any(isnan(Data.ArrDelay),2),:);
Data(any(Data.ArrDelay<0,2),:)=[];
nanmean(Data.ArrDelay);

% The dataset consists of flight arrival and departure times for every 
% commercial fight in the USA from January 2008 to April 2008. This dataset 
% contains extensive information about almost 2 million flights, including 
% the delay (in minutes) in reaching the destination. The average delay of 
% a flight in the first 4 months of 2008 was of 30 minutes.
% We chose to include into our model 8 of the many variables available for 
% this dataset: the age of the aircraft (number of years since deployment), 
% distance that needs to be covered, airtime, departure time, arrival time, 
% day of the week, day of the month and month (Hensman et al, 2013).
vars = {'Month','DayofMonth','DayOfWeek','DepTime','ArrTime','AirTime',...
    'Distance','AircraftAge','ArrDelay'};
Air_Data = Data(:,vars); % Keep only the variables in Hensman's paper.
whos Air_Data
Air_Data.DepTime = str2double(Air_Data.DepTime);
Air_Data.ArrTime = str2double(Air_Data.ArrTime);
Air_Data.AirTime = str2double(Air_Data.AirTime);

% Transform the table to matrix
X = table2array(Air_Data);

% Airplane Age
% One issue was with the airplane age, where in some cases the age was
% either 0 either 2008. We replaced these 'abnormal' values with the 
% average age. Nevertheless, because the cases were so few it didn't 
% really change anything on the final results.
nanmean(X(:,8)); % Check the mean without the NaN values
X(isnan(X(:,8)),8) = 13; % Replace the NAN values with the mean above
X((X(:,8)==2008),8) = 13; % There are some dates that are 2008
X((X(:,8)==0),8) = 13; % There are some dates that are 0
max(X(:,8));
min(X(:,8));

% Check if the data in the columns agree with real life data.
min(X(:,1)); max(X(:,1)); % Month 1-12
min(X(:,2)); max(X(:,2)); % DayofMonth 1-31
min(X(:,3)); max(X(:,3)); % DayOfWeek 1 (Monday) - 7 (Sunday)
min(X(:,4)); max(X(:,4)); % DepTime hhmm 0001 - 2400
min(X(:,5)); max(X(:,5)); % ArrTime hhmm 0001 - 2400
min(X(:,6)); max(X(:,6)); % AirTime minutes 0
X(any(X(:,6)<15,2),:)=[]; % Drop flights bellow 15 mins airtime
min(X(:,7)); max(X(:,7)); % Distance miles

% Because mean(y) is not zero, and we want to use a zero mean GP
% we have to centralise it.
%offset = mean(X(:,9));
%std(X(:,9));
%X(:,9) = X(:,9)- offset;
%mean(X(:,9));
% Or normalise everything instead (that's what we did).
X = zscore(X);

