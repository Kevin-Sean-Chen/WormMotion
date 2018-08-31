% prelimMNRplots
clear all;close all

Ldfile = '/users/mikio/dropbox/wormdata/heatData20141212';
% Ldfile = 'heatData20141214';
% Ldfile = 'heatData20150118';
load(Ldfile);
%% Interpolate behavior and missing data

% % Align time points from ethogram with flourescence
% figure;plot(hiResFrameTime,ethogram,hasPointsTime,interpEtho,'--')
% title('interpolated ethogram')

%interpolate missing data
intG2 = interp1(hasPointsTime,G2',hasPointsTime,'spline');
intR2 = interp1(hasPointsTime,R2',hasPointsTime,'spline');

gmax = max(max(G2));
gmin = min(min(G2));

%exclude observations with nan in behavior array
interpEtho = round(interp1(hiResFrameTime,ethogram,hasPointsTime,'next'));
indNan = find(isnan(interpEtho));
interpEtho(indNan) = [];
intG2(indNan,:) = [];
intR2(indNan,:) = [];

hasPointsTime(indNan) = [];


% plot of behavior 1 = forward, 0 = turn, -1 = reverse
indminus = find(interpEtho==-1);
interpEtho(indminus) = 2*ones(length(indminus),1);%reverse
indzero = find(interpEtho==0);
interpEtho(indzero) = 3*ones(length(indzero),1);%turn

[nt n] = size(intG2);
%% Set training and testing periods for multinomial regression
fracdata  = 1;
trnind = 1:round(fracdata*nt);
tstind = setdiff(1:nt,trnind);
%% Plot for GCamp6
[B,dev,stats] = mnrfit(intG2(trnind,:),interpEtho(trnind,:),'model','nominal');
pihat = mnrval(B,intG2);

figure;
imagesc(hasPointsTime,[0.5 .5],interpEtho');set(gca,'ydir','normal')
colormap bone
hold on
plot(hasPointsTime,pihat(:,1),hasPointsTime,pihat(:,2),hasPointsTime,pihat(:,3),'linewidth',2);
plot([hasPointsTime(round(fracdata*nt)) hasPointsTime(round(fracdata*nt))],[0 1],'g','linewidth',2)
hold off
legend('forward','reverse','turn');
title('Predicted behavior GCaMP6s (1=forward, 2=reverse, 3=turn)')
colorbar
xlabel('time');ylabel('Probability of behavior')

%% Plot for RFP
[B,dev,stats] = mnrfit(intR2(trnind,:),interpEtho(trnind,:),'model','nominal');

% lnodds = repmat(B(1,:)',1,length(hasPointsTime)) + B(2:end,:)'*intR2';

pihat = mnrval(B,intR2);

figure;
imagesc(hasPointsTime,[0.5 .5],interpEtho');set(gca,'ydir','normal')
colormap bone
hold on
plot(hasPointsTime,pihat(:,1),hasPointsTime,pihat(:,2),hasPointsTime,pihat(:,3),'linewidth',2);
plot([hasPointsTime(round(fracdata*nt)) hasPointsTime(round(fracdata*nt))],[0 1],'g','linewidth',2)
hold off
legend('forward','reverse','turn');
title('Predicted behavior RFP (1=forward, 2=reverse, 3=turn)')
colorbar
xlabel('time');ylabel('Probability of behavior')