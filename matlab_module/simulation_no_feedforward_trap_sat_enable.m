
%% Generate Command (Trapezoidal velocity)
[pos_cmd, vel_cmd, acc_cmd, t] = generate_cmd(start_position, goal_position, Vmax, Amax, sampT);
cmd = [t; pos_cmd]';
vel_cmd = [t; vel_cmd]';
acc_cmd = [t; acc_cmd]';

total_time = length(t)*sampT;

%% Simulation
close_position = 1;
close_velocity = 1;
close_friction = 1;

sim("motor_no_feedforward_Kvi_sat_enable_T_load.slx")


%% For aligm the length of each array
% pos, pos_cmd, pos_error
pos_cmd = pos_cmd';
tout = tout(1:length(pos_cmd));
pos = pos(1:length(pos_cmd));
vel = vel(1:length(pos_cmd));
pos_error = pos_error(1:length(pos_cmd));
torque = torque(1:length(pos_cmd));
% friction = friction(1:length(pos_cmd));

%% Function
function [P, V, A, t] = generate_cmd(start, goal, Vmax, Amax, sampT)
    % 計算時間
    dis = goal - start;                  % 目標位置和起始位置的距離
    t1 = Vmax / Amax;                    % 加速階段時間
    t2 = (dis - t1 * Vmax) / Vmax;       % 等速階段時間
    total_time = 2 * t1 + t2 + 1.0;      % 總時間

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