
%% Generate Command (Trapezoidal velocity)
[pos_cmd, vel_cmd, acc_cmd, t] = generate_cmd(start_position, goal_position, Vmax, Amax, sampT);
pos_cmd = ones(1, length(pos_cmd));
cmd = [t; pos_cmd]';
vel_cmd = [t; vel_cmd]';
acc_cmd = [t; acc_cmd]';

% total_time = length(t)*sampT;
total_time = 1.5;

%% 雙質點模型參數設定
% J1 = 0.195e-3;    % 馬達端慣量
% J = J1;
% 
% freq_r  = 550;     % 共振頻率(Hz)
% freq_ar = 500;    % 反共振頻率(Hz)
% zeta    = 0.001;     % 阻尼係數
% 
% omega_r  = 2 * pi * freq_r; % 共振頻率(Hz)
% omega_ar = 2 * pi * freq_ar; % 反共振頻率(Hz)
% 
% J2 = (omega_r^2*J1/omega_ar^2) - J1;
% meff = (J1 * J2) / (J1 + J2); % 等效質量
% 
% K12 = omega_ar^2 * J2;
% C12 = zeta*(2*sqrt(meff*K12));


%% GM, PM
close_position = 1; % Initialize flag
close_velocity = 1; % Initialize flag
close_friction = 1;

try
    close_position = 0;
    % io = getlinio('motor_no_feedforward_Kvi');
    % sys_position = linearize('motor_no_feedforward_Kvi', [io(1), io(5)]);
    % [Gm_position, Pm_position, ~, ~] = margin(sys_position);
    % Gm_position = 20 * log10(Gm_position);
    
    close_velocity = 0;
    close_friction = 0;
    io = getlinio('motor_no_feedforward_Kvi');
    sys_velocity = linearize('motor_no_feedforward_Kvi', [io(1), io(2)]);
    [Gm_velocity, Pm_velocity, ~, ~] = margin(sys_velocity);
    Gm_velocity = 20 * log10(Gm_velocity);
catch ME
    Gm_position = -10; 
    Pm_position = 0;
    Gm_velocity = -10;
    Pm_velocity = 0;
end
%% Simulation
close_position = 1;
close_velocity = 1;
close_friction = 1;

sim("motor_no_feedforward_Kvi.slx")


%% For aligm the length of each array
% pos, pos_cmd, pos_error
pos_cmd = pos_cmd';
tout = tout(1:length(pos_cmd));
pos = pos(1:length(pos_cmd));
vel = vel(1:length(pos_cmd));
pos_error = pos_error(1:length(pos_cmd));
torque = torque(1:length(pos_cmd));
% friction = friction(1:length(pos_cmd));

% pos_cmd = ones(length(pos_cmd));

%% Function
function [P, V, A, t] = generate_cmd(start, goal, Vmax, Amax, sampT)
    % 計算時間
    dis = goal - start;                  % 目標位置和起始位置的距離
    t1 = Vmax / Amax;                    % 加速階段時間
    t2 = (dis - t1 * Vmax) / Vmax;       % 等速階段時間
    total_time = 2 * t1 + t2 + 0.5;      % 總時間

    % 根據採樣時間 sampT 來生成時間範圍
    t = 0:sampT:total_time;              % 使用步長為 sampT 的時間範圍

    % 預分配加速度、速度和位置向量
    A = zeros(size(t));  
    V = zeros(size(t));  
    P = zeros(size(t));  

    % 根據時間分段計算加速度、速度和位置
    for i = 1:length(t)
        if t(i) <= t1
            % 加速階段
            A(i) = Amax;
            V(i) = Amax * t(i);
            P(i) = start + (1/2) * Amax * t(i)^2;
        elseif t(i) <= t1 + t2
            % 恆速階段
            A(i) = 0;
            V(i) = Vmax;
            P(i) = start + (1/2) * Amax * t1^2 + Vmax * (t(i) - t1);
        elseif t(i) <= 2 * t1 + t2
            % 減速階段
            A(i) = -Amax;
            V(i) = Vmax - Amax * (t(i) - (t1 + t2));
            P(i) = start + (1/2) * Amax * t1^2  ...
                         + Vmax * t2 ...
                         + Vmax * (t(i) - (t1 + t2)) ...
                         - (1/2) * Amax * (t(i) - (t1 + t2))^2;
        else
            % 最終位置
            P(i) = start + (1/2) * Amax * t1^2 + Vmax * t2 + Vmax * (2 * t1 + t2 - (t1 + t2)) - (1/2) * Amax * (2 * t1 + t2 - (t1 + t2))^2;
            V(i) = 0;
            A(i) = 0;
        end
    end
end