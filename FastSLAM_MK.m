clear;
close all;
clc;

global dt;
time    = 0;
endtime = 60; % [sec]
dt      = 0.1; % [sec]
 
nSteps = ceil((endtime - time)/dt);
 
result.time  = [];
result.xTrue = [];
result.xEst  = [];
result.mu    = [];

% State Vector [x y yaw]'
xEst = [0 0 0]';
 
% True State
xTrue = xEst;
 
% Covariance Matrix for predict
Q = diag([0.1 0.1 toRadian(3)]).^2;
 
% Covariance Matrix for observation
R = diag([10 toRadian(30)]).^2; % range[m], Angle[rad]

% Simulation parameter
global Qsigma
Qsigma = diag([0.1 toRadian(5)]).^2;
 
global Rsigma
Rsigma = diag([0.1 toRadian(1)]).^2;

%LM�^�O�̈ʒu [x, y]
% LM = [10 0;
%       10 10;
%       0 15;
%       10 25;
%       0 25;
%      -5 20];
LM = [10 5;
      3 9;
      2 5;
      8 15;
      10 25;
      5 25;
      0 30;
      -10 25;
      -5 15;
      0 20;
      -5 25;
      -15 10];

MAX_RANGE = 30;%�ő�ϑ�����
NP        = 100;%�p�[�e�B�N����
NTh       = NP / 2.0;%���T���v�����O�����{����L���p�[�e�B�N����

% % �����h�}�[�N�ʒu�ƑΉ��ϐ�
% LM = zeros(30, 2);
% LM(:, 1:2) = -20 + (20 + 20) * rand(30, 2);
 
px = repmat(xEst, 1, NP);%�p�[�e�B�N���i�[�ϐ�
pw = zeros(1, NP) + 1 / NP;%�d�ݕϐ�
% �ϑ��l�ɑ΂���flag
flag = zeros(size(LM, 1), NP);
% �e�p�[�e�B�N�����������h�}�[�N�ʒu
mu = zeros(2 * NP, size(LM, 1));
% �e�p�[�e�B�N�����������h�}�[�N�����U�s��
Sigma = 100000 * ones(2 * NP, 2 * size(LM, 1));

tic;
% Main loop
for i = 1 : nSteps
    time = time + dt;
    % Input
    u = doControl(time);
    % Observation
    [z, xTrue, u] = Observation(xTrue, u, LM, MAX_RANGE);
    % ------ Particle Filter --------
    for ip = 1 : NP
        x = px(:, ip);
        w = pw(ip);
        % Dead Reckoning and random sampling
        x = f(x, u) + sqrt(Q) * randn(3,1);
        % Calc Inportance Weight
        for iz = 1:length(z(:, 1))
            if (flag(iz, ip) < 1.0)
                % LM�����߂Ċϑ�����ꍇ
                % �ϑ��l�̕��ϒl�̏�����
                mu(3 * ip - 2:3 * ip - 1, iz) = xEst(1 : 2) + [z(iz, 1) * cos(PI2PI(xEst(3) + z(iz, 2))); z(iz, 1) * sin(PI2PI(xEst(3) + z(iz, 2)))]; % (8.44)��
                % ���R�r�A��H�s��
                H = jacobian_H(mu, x);
                % �ϑ��l�̋����U�s�񏉊���
                Sigma(3 * ip - 2:3 * ip - 1, 3 * iz - 2:3 * iz - 1) = inv(H' / R * H); %(8,48)��
                flag(iz, ip)     = 1.0;
            else
                % LM�̊ϑ������߂Ăł͂Ȃ��ꍇ
                [expectedZ, H] = measurement_model(mu(3 * ip - 2:3 * ip - 1, iz), x);
                % �v���̋����U
                Rt = (H * Sigma(3 * ip - 2:3 * ip - 1, 3 * iz - 2:3 * iz - 1) * H') + R; % (8.38)���̊��ʂ̒�
                % �J���}���Q�C���̌v�Z
                K = (Sigma(3 * ip - 2:3 * ip - 1, 3 * iz - 2:3 * iz - 1) * H') / Rt; % (8.38)��
                % ���ϒl�̍X�V
                error         = [z(iz, 1) - expectedZ(1); z(iz, 2) - expectedZ(2)];
                mu(3 * ip - 2:3 * ip - 1, iz) = mu(3 * ip - 2:3 * ip - 1, iz) + K * error; %(8.37)��
                % �����U�̍X�V
                Sigma(3 * ip - 2:3 * ip - 1, 3 * iz - 2: 3 * iz - 1) = (eye(2) - K * H) * Sigma(3 * ip - 2:3 * ip - 1, 3 * iz - 2:3 * iz - 1); % (8.40)��
                w                = w * likelihood(error, Rt);
            end
        end
        px(:, ip) = x;%�i�[
        pw(ip)    = w;
    end
    pw       = Normalize(pw, NP);%���K��
    [px, pw] = Resampling(px, pw, NTh, NP);%���T���v�����O
    xEst     = px * pw';%�ŏI����l�͊��Ғl
    xEst(3)  = PI2PI(xEst(3));%�p�x�␳
    
    % Simulation Result
    result.time  = [result.time; time];
    result.xTrue = [result.xTrue; xTrue'];
    result.xEst  = [result.xEst;xEst'];
end
toc

DrawGraph(result, LM, mu);

% function [px_res, pw_res] = Resampling( px, pw, Nth, NP)
%     Neff   = 1.0 / (pw * pw');
%     if Neff < Nth
%         px_tmp = px;
%         pw     = pw / sum(pw);
%         pw_cdf = cumsum(pw);
%         for j = 1 : NP
%             index_find = find(rand <= pw_cdf, 1);
%             px_tmp(:, j) = px(:, index_find);
%         end
%         px_res = px_tmp;
%         pw_res = ones(1, NP) / NP;
%     else
%         px_res = px;
%         pw_res = pw;
%     end
% end

function [px_res, pw_res] = Resampling(px, pw, Nth, NP)
    Neff   = 1.0 / (pw * pw');
    if Neff < Nth
        pw_cdf  = cumsum(pw);
        px_temp = px;
        r       = rand / NP;
        for n = 1 : NP
            index = find(pw_cdf >= r, 1);
            px_temp(:, n) = px(:, index);
            r = r + 1 / NP;
        end
        px_res = px_temp;
        pw_res = ones(1, NP) / NP;
    else
        px_res = px;
        pw_res = pw;
    end
end

function pw=Normalize(pw,NP)
    %�d�݃x�N�g���𐳋K������֐�
    sumw=sum(pw);
    if sumw~=0
        pw=pw/sum(pw);%���K��
    else
        pw=zeros(1,NP)+1/NP;
    end
end

function p = likelihood(error, sigma)
    %�K�E�X���z�̊m�����x���v�Z����֐�
    p = (1 / (sqrt(2 * pi * det(sigma))) * exp(- 1 / 2 * error' / sigma * error));
end

function x = f(x, u)
    % Motion Model
    global dt;
    if u(2) == 0
        F = [u(1) * dt * cos(x(3))
             u(1) * dt * sin(x(3))
             u(2) * dt];
    else
        F = [u(1) / u(2) * (sin(x(3) + u(2) * dt) - sin(x(3)))
             u(1) / u(2) * (-cos(x(3) + u(2) * dt) + cos(x(3)))
             u(2) * dt];
    end
    x    = x + F;
    x(3) = PI2PI(x(3));%�p�x�␳
end

function u = doControl(time)
    %Calc Input Parameter
    T=10; % [sec]
    % [V yawrate]
    V=1.0; % [m/s]
    yawrate = 5; % [deg/s]
%     
%     if time > 10 && time <= 15
%         u = [V; pi/10.0];
%     elseif time > 15 && time <= 30
%         u = [V; 0.0];
%     elseif time > 30 && time <= 35
%         u = [V; pi/10.0];
%     elseif time > 35 && time <= 50
%         u = [V; 0.0];
%     elseif time > 50 &&time <= 55
%         u = [V; pi/10.0];
%     else
%         u = [V; 0.0];
%     end     
    
    u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';
end

function H = jacobian_H(z, x)
    % ���R�r�A���s��H
    H = [(z(3) - x(1)) / z(1) (z(4) - x(2)) / z(1);
         (x(2) - z(4)) / z(1)^2 (z(3) - x(1)) / z(1)^2];
end

function [expectedZ, H] = measurement_model(mu, x)
    dx        = mu(1) - x(1);
    dy        = mu(2) - x(2);
    distance  = sqrt(dx^2 + dy^2);
    theta     = atan2(dy, dx) - x(3);
    expectedZ = [distance; PI2PI(theta)];
    H         = [(mu(1) - x(1)) / expectedZ(1) (mu(2) - x(2)) / expectedZ(1);
                 (x(2) - mu(2)) / expectedZ(1)^2 (mu(1) - x(1)) / expectedZ(1)^2];
end

%Calc Observation from noise prameter
function [z, x, u] = Observation(x, u, LM, MAX_RANGE)
    global Qsigma Rsigma;
    
    x = f(x, u);% Ground Truth
    u = u + sqrt(Qsigma) * randn(2, 1);%add Process Noise
    %Simulate Observation
    z = [];
    for iz = 1 : length(LM(:, 1))
        dx       = LM(iz, 1) - x(1);
        dy       = LM(iz, 2) - x(2);
        distance = sqrt(dx^2 + dy^2);
        theta    = atan2(dy, dx) - x(3);
        if distance < MAX_RANGE %�ϑ��͈͓�
            noise = Rsigma * randn(2, 1);
            z = [z; [distance + noise(1) PI2PI(theta + noise(2)) LM(iz, :)]];
        end
    end
end

function angle=PI2PI(angle)
    %���{�b�g�̊p�x��-pi~pi�͈̔͂ɕ␳����֐�
    angle = mod(angle, 2*pi);

    i = find(angle > pi);
    angle(i) = angle(i) - 2*pi;

    i = find(angle < -pi);
    angle(i) = angle(i) + 2*pi;
end

function []=DrawGraph(result, LM, mu)
    %Plot Result
    figure(1);
    hold off;
    x=[result.xTrue(:,1:2) result.xEst(:,1:2)];
    set(gca, 'fontsize', 16, 'fontname', 'times');
    plot(x(:,1), x(:,2),'-.b','linewidth', 4); hold on;
    plot(x(:,3), x(:,4),'r','linewidth', 4); hold on;
    plot(LM(:, 1), LM(:, 2), 'pentagram', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'blue'); hold on;
    for i = 1 : size(LM, 1)
        plot(mu(298, i), mu(299, i), 'pentagram', 'MarkerSize', 15); hold on;
    end
    title('PF Localization Result', 'fontsize', 16, 'fontname', 'times');
    xlabel('X (m)', 'fontsize', 16, 'fontname', 'times');
    ylabel('Y (m)', 'fontsize', 16, 'fontname', 'times');
    legend('Ground Truth','PF', 'LandMark', 'Estimated LandMark');
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