
%% Generate Command (Step command)
total_time = 3.0;
pos_cmd = ones(1, total_time*1000); % total_time sec
cmd = [linspace(0.0, total_time, length(pos_cmd)); pos_cmd]';

%% GM, PM
close_position = 1; % Initialize flag
close_velocity = 1; % Initialize flag
close_friction = 1;

try
    close_position = 0;
    % io = getlinio('motor_no_feedforward_Kvi_sat_T_load');
    % sys_position = linearize('motor_no_feedforward_Kvi_sat_T_load', [io(1), io(5)]);
    % [Gm_position, Pm_position, ~, ~] = margin(sys_position);
    % Gm_position = 20 * log10(Gm_position);
    
    close_velocity = 0;
    close_friction = 0;
    io = getlinio('motor_no_feedforward_Kvi_sat_T_load');
    sys_velocity = linearize('motor_no_feedforward_Kvi_sat_T_load', [io(1), io(2)]);
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

if (Kvi==0.0)
    gain_anti_windup = 0.0;
else
    gain_anti_windup = 0.05 / (Kvi * sampT);
end

% sim("motor_no_feedforward_Kvi_sat_T_load.slx")
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
