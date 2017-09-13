%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots
% 1. GP function plot for inducing points
% We plot the predicted means over the x values
id = linspace(1,length(muA),length(muA))';
rand_s = randsample(id,60);
mu = mu50(rand_s);
mub = mu100(rand_s);
muc = mu250(rand_s);
mud = mu500(rand_s);
s2 = s50(rand_s);
s2B = s100(rand_s);
s2C = s250(rand_s);
s2D = s500(rand_s);
fA = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fB = [mub+2*sqrt(s2B); flipdim(mub-2*sqrt(s2B),1)];
fC = [muc+2*sqrt(s2C); flipdim(muc-2*sqrt(s2C),1)];
fD = [mud+2*sqrt(s2D); flipdim(mud-2*sqrt(s2D),1)];
xs = linspace(1,60,60)';
subplot(221)
fill([xs; flipdim(xs,1)], fA, [7 7 7]/8)
hold on; plot(xs, mu, 'b'); title('Subplot 1: $u=50$','interpreter','latex'); 
ylabel('$f(x)$','interpreter','latex');
subplot(222)
fill([xs; flipdim(xs,1)], fB, [7 7 7]/8)
hold on; plot(xs, mub, 'b'); title('Subplot 2: $u=100$','interpreter','latex'); 
subplot(223)
fill([xs; flipdim(xs,1)], fC, [7 7 7]/8)
hold on; plot(xs, muc, 'b'); title('Subplot 3: $u=250$','interpreter','latex'); 
ylabel('$f(x)$','interpreter','latex'); xlabel('$x$','interpreter','latex');
subplot(224)
fill([xs; flipdim(xs,1)], fD, [7 7 7]/8)
hold on; plot(xs, mud, 'b'); title('Subplot 4: $u=500$','interpreter','latex'); 
xlabel('$x$','interpreter','latex');

% 2. MSE/inducing points
% We plot the average MSE of the two inducing point methods VFE and FITC 
% over the 4 inducing points values (50, 100, 250, 500).
vars = {'50', '100', '250', '500'};
inducing_p = linspace(1,4,4);
ind_table_mse = [mean(table50(:,1)) mean(table100(:,1)) mean(table250(:,1)) mean(table500(:,1))];
ind_table_mseF = [mean(tableF50(:,1)) mean(tableF100(:,1)) mean(tableF250(:,1)) mean(tableF500(:,1))];
plot(inducing_p, ind_table_mse);
hold on;
plot(inducing_p, ind_table_mseF);
title('ASA Flight Delay Dataset','interpreter','latex');
xlabel('Inducing points','interpreter','latex');
ylabel('MSE','interpreter','latex');
legend('VFE','FITC','Location','northeast');
set(gca,'XTick',[1 2 3 4]); 
set(gca,'XTickLabel',vars);
xtickangle(45);

% 3. time/inducing points
% We plot the run-time of the algorithm for the two methods VFE and FITC
% over the inducing points values
vars = {'50', '100', '250', '500'};
inducing_p = linspace(1,4,4);
ind_table_time = [mean(table50(:,3)) mean(table100(:,3)) mean(table250(:,3)) mean(table500(:,3))];
ind_table_timeF = [mean(tableF50(:,3)) mean(tableF100(:,3)) mean(tableF250(:,3)) mean(tableF500(:,3))];
plot(inducing_p, ind_table_time);
hold on;
plot(inducing_p, ind_table_timeF);
title('ASA Flight Delay Dataset','interpreter','latex');
xlabel('Inducing points','interpreter','latex');
ylabel('Run time (s)','interpreter','latex');
legend('VFE','FITC','Location','northwest');
set(gca,'XTick',[1 2 3 4]); 
set(gca,'XTickLabel',vars);
xtickangle(45);

% 4. exp(hyp)
% We plot the inverse lengthscales from the SD optimization (hyp2)
vars = {'Month','DayofMonth','DayOfWeek','DepTime','ArrTime','AirTime',...
    'Distance','AircraftAge'};
exp_l = 1./exp(hyp2.cov);
v = linspace(1,8,8);
bar(v,exp_l(:,1:8));
set(gca,'XTickLabel',{'Month','DayofMonth','DayOfWeek','DepTime','ArrTime',...
    'AirTime','Distance','AircraftAge'});
xtickangle(45);
ylabel('$Inverse \quad lengthscale$','interpreter','latex');

% 5. Mini-batches (SD method) boxplot
% We plot the mean predicted values of the SD (50, 100, 250, 500, 1000,
% 1700). For each inference method (exact, Laplace, VB).
% Exact Inf
z = [Table1(:,1); Table2(:,1); Table3(:,1); Table4(:,1)];
zg = [zeros(length(Table1(:,1)),1); ones(length(Table2(:,1)),1); 2*ones(length(Table3(:,1)),1); 3*ones(length(Table4(:,1)),1)];
boxplot(z,zg, 'Labels',{'mu = .89 (250)',...
    'mu = .87 (500)','mu = .87 (1000)', 'mu = .86 (1700)'});
title('$Exact \quad Inference$','interpreter','latex'); 
xlabel('mean (mini-batch)','interpreter','latex');
ylabel('MSE','interpreter','latex');

% Laplace
z = [LTable1(:,1); LTable2(:,1); LTable3(:,1); LTable4(:,1)];
zg = [zeros(length(LTable1(:,1)),1); ones(length(LTable2(:,1)),1); 2*ones(length(LTable3(:,1)),1); 3*ones(length(LTable4(:,1)),1)];
boxplot(z,zg, 'Labels',{'mu = .89 (250)',...
    'mu = .87 (500)','mu = .86 (1000)', 'mu = .84 (1700)'});
title('$Laplace \quad Approximation$','interpreter','latex'); 
xlabel('mean (mini-batch)','interpreter','latex');
ylabel('MSE','interpreter','latex');

% VB
z = [VBTable1(:,1); VBTable2(:,1); VBTable3(:,1); VBTable4(:,1)];
zg = [zeros(length(VBTable1(:,1)),1); ones(length(VBTable2(:,1)),1); 2*ones(length(VBTable3(:,1)),1); 3*ones(length(VBTable4(:,1)),1)];
boxplot(z,zg, 'Labels',{'mu = .89 (250)',...
    'mu = .87 (500)','mu = .88 (1000)', 'mu = .87 (1700)'});
title('$VB \quad Approximation$','interpreter','latex'); 
xlabel('mean (mini-batch)','interpreter','latex');
ylabel('MSE','interpreter','latex');
  
% 6. MSE/runs
% We plot the average MSE per run for the SD method.
figure
subplot(311); % Exact
y_label = linspace(0,12,13);
plot(Table3(:,1)); hold on 
ylabel({'MSE','Exact Inference'},'interpreter','latex');
title('Compare Mean Squared Error for different inference methods',...
    'interpreter','latex');
subplot(312); % Laplace
plot(LTable3(:,1));
ylabel({'MSE','Laplace'},'interpreter','latex');
subplot(313); % VB
plot(VBTable3(:,1));
xlabel('Test-runs','interpreter','latex');
ylabel({'MSE','VB'},'interpreter','latex');

% 7. MSE/runs
% The same as [6] but the average MSE were drawn on the same plot.
figure
plot(Table3(:,1)); hold on
title('Compare Mean Squared Error for different inference methods',...
    'interpreter','latex');
plot(LTable3(:,1));
plot(VBTable3(:,1));
legend('Exact Inference','Laplace','VB','Location','northwest');
xlabel('Test-runs','interpreter','latex');
ylabel('MSE','interpreter','latex');

% 8. time/batch size
% We plot the average runtime by batch-size (SD method).
table_time = [p1 Lp1 VBp1; p2 Lp2 VBp2; p3 Lp3 VBp3; p4 Lp4 VBp4];
bar(table_time);
title('ASA Flight Delay Dataset','interpreter','latex');
legend('EI','L', 'VB','Location','northwest');
xlabel('mini-batch','interpreter','latex');
ylabel('Run time $(s)$','interpreter','latex');
set(gca,'XTickLabel',{'250','500','1000','1700'});
xtickangle(45);
