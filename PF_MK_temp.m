clear;
close all;
clc;
 
time = 0;
endtime = 60; % [sec]
global dt;
dt = 0.1; % [sec]
 
nSteps = ceil((endtime - time)/dt);
 
result.time=[];
result.xTrue=[];
result.xEst=[];

% State Vector [x y yaw]'
xEst=[0 0 0]';
 
% True State
xTrue=xEst;
 
% Covariance Matrix for predict
Q=diag([0.1 0.1 toRadian(3)]).^2;
 
% Covariance Matrix for observation
R=diag([1]).^2;%[range [m]

% Simulation parameter
global Qsigma
Qsigma=diag([0.1 toRadian(5)]).^2;
 
global Rsigma
Rsigma=diag([0.1]).^2;

%RFIFタグの位置 [x, y]
RFID=[10 0;
      10 10;
      0  15
      -5 20];
  
MAX_RANGE=20;%最大観測距離
NP=100;%パーティクル数
NTh=NP/2.0;%リサンプリングを実施する有効パーティクル数

px=repmat(xEst,1,NP);%パーティクル格納変数
pw=zeros(1,NP)+1/NP;%重み変数
 
tic;
%movcount=0;
% Main loop
for i=1 : nSteps
    time = time + dt;
    % Input
    u=doControl(time);
    % Observation
    [z,xTrue, u]=Observation(xTrue, u, RFID, MAX_RANGE);
    
    % ------ Particle Filter --------
    for ip=1:NP
        x=px(:,ip);
        w=pw(ip);
        % Dead Reckoning and random sampling
        x=f(x, u)+sqrt(Q)*randn(3,1);
        % Calc Inportance Weight
        for iz=1:length(z(:,1))
            pz=norm(x(1:2)'-z(iz,2:3));
            dz=pz-z(iz,1);
            w=w*Gauss(dz,0,sqrt(R));
        end
        px(:,ip)=x;%格納
        pw(ip)=w;
    end
    
    pw=Normalize(pw,NP);%正規化
    [px,pw]=Resampling(px,pw,NTh,NP);%リサンプリング
    xEst=px*pw';%最終推定値は期待値
    
    % Simulation Result
    result.time=[result.time; time];
    result.xTrue=[result.xTrue; xTrue'];
    result.xEst=[result.xEst;xEst'];
end
toc

DrawGraph(result);

function [px_res, pw_res] = Resampling( px, pw, Nth, NP)
    Neff   = 1.0 / (pw * pw');
    if Neff < Nth
        px_tmp = px;
        pw     = pw / sum(pw);
        pw_cdf = cumsum(pw);
        for j = 1 : NP
            index_find = find(rand <= pw_cdf, 1);
            px_tmp(:, j) = px(:, index_find);
        end
        px_res = px_tmp;
        pw_res = ones(1, NP) / NP;
    else
        px_res = px;
        pw_res = pw;
    end
end

function pw=Normalize(pw,NP)
    %重みベクトルを正規化する関数
    sumw=sum(pw);
    if sumw~=0
        pw=pw/sum(pw);%正規化
    else
        pw=zeros(1,NP)+1/NP;
    end
end
    

function p=Gauss(x,u,sigma)
    %ガウス分布の確率密度を計算する関数
    p=1/sqrt(2*pi*sigma^2)*exp(-(x-u)^2/(2*sigma^2));
end

function x = f(x, u)
    % Motion Model
    
    global dt;
    
    F = [1 0 0
        0 1 0
        0 0 1];
    
    B = [
        dt*cos(x(3)) 0
        dt*sin(x(3)) 0
        0 dt];
    
    x= F*x+B*u;
end

function u = doControl(time)
    %Calc Input Parameter
    T=10; % [sec]
    
    % [V yawrate]
    V=1.0; % [m/s]
    yawrate = 5; % [deg/s]
    
    u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';
end

%Calc Observation from noise prameter
function [z, x, u] = Observation(x, u, RFID,MAX_RANGE)
    global Qsigma;
    global Rsigma;
    
    x=f(x, u);% Ground Truth
    u=u+sqrt(Qsigma)*randn(2,1);%add Process Noise
    %Simulate Observation
    z=[];
    for iz=1:length(RFID(:,1))
        d=norm(RFID(iz,:)-x(1:2)');
        if d<MAX_RANGE %観測範囲内
            z=[z;[d+sqrt(Rsigma)*randn(1,1) RFID(iz,:)]];
        end
    end
end

function []=DrawGraph(result)
    %Plot Result
    
    figure(1);
    hold off;
    x=[ result.xTrue(:,1:2) result.xEst(:,1:2)];
    set(gca, 'fontsize', 16, 'fontname', 'times');
    plot(x(:,1), x(:,2),'-.b','linewidth', 4); hold on;
    plot(x(:,3), x(:,4),'r','linewidth', 4); hold on;
    
    title('PF Localization Result', 'fontsize', 16, 'fontname', 'times');
    xlabel('X (m)', 'fontsize', 16, 'fontname', 'times');
    ylabel('Y (m)', 'fontsize', 16, 'fontname', 'times');
    legend('Ground Truth','PF');
    grid on;
    axis equal;
end

function radian = toRadian(degree)
    % degree to radian
    radian = degree/180*pi;
end

function degree = toDegree(radian)
    % radian to degree
    degree = radian/pi*180;
end