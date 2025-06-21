close all; clear; clc;
%% Simulink檔名
model_name = 'Two_mass_sim.slx';
addpath('utils');
%% 雙質點模型參數設定
J1 = 0.195e-3; % 馬達端慣量

freq_r = 550; % 共振頻率(Hz)
freq_ar = 500; % 反共振頻率(Hz)
zeta = 0.001; % 阻尼係數

omega_r = 2 * pi * freq_r; % 共振頻率(Hz)
omega_ar = 2 * pi * freq_ar; % 反共振頻率(Hz)

%% 其他設定
% Command Input --- 透過S curve生成位置、速度、加速度命令
Initial = 0;
Final = 5;
sampling_t = 0.001;
a_avg = 0.75;
VelLimit = 10;
AccLimit = 50;
[ JointCmd , Time ] = Scurve_MultiAxis ( Initial , Final , sampling_t , a_avg , VelLimit, AccLimit);
static_count = 200;
cmd_pos = [JointCmd(:, 1); ones(static_count, 1) * JointCmd(end, 1)];
cmd_spd = [JointCmd(:, 2); ones(static_count, 1) * JointCmd(end, 2)];
time = [Time, Time(end)+sampling_t:sampling_t:Time(end)+sampling_t * static_count]';

% Parameter Setting --- 設定控制器參數
BW = 100;
KPP = BW * 0.5 * pi;
KVP = BW * 2 * pi;
Jm = J1;

%% 計算雙質量模型其他參數 J2、C12、K12
J2 = (omega_r^2*J1/omega_ar^2) - J1;
meff = (J1 * J2) / (J1 + J2); % 等效質量

K12 = omega_ar^2 * J2;
C12 = zeta*(2*sqrt(meff*K12));

% 轉移函數建構 (簡單驗證)
s = tf('s');
system = ((J2 * s^2) + K12 + C12 * s) / (J1 * J2 * s^4 + (J1 + J2) * C12 * s^3 + (J1 + J2) * K12 * s^2);

% 顯示轉移函數
disp('轉移函數 G(s):');
system

% 共振頻率與反共振頻率
omega_r_ = sqrt(K12 / meff);
omega_ar_ = sqrt(K12 / J2);

% 阻尼係數
zeta_ = C12 / (2 * sqrt(meff * K12));

% 顯示計算結果
fprintf('共振頻率 (Hz): %.4f\n', omega_r_ / (2 * pi));
fprintf('反共振頻率 (Hz): %.4f\n', omega_ar_ / (2 * pi));
fprintf('阻尼係數: %.4f\n', zeta_);

% Bode 圖檢視
figure;
bode(system);
grid on;
title('雙質點系統 Bode 圖');

%% bode plot by Simulink
io = getlinio('Two_mass_sim');
sys_simulink = linearize('Two_mass_sim', [io(1), io(2)]);
% [Gm_velocity, Pm_velocity, ~, ~] = margin(sys_velocity);
% Gm_velocity = 20 * log10(Gm_velocity);
frequency_bode = 10.^(3:0.01:4);
figure
margin(sys_simulink, frequency_bode)

%% 模擬結果
Tf = time(end);
sim(model_name,[0 Tf]);

figure();
plot(time, cmd_pos);
hold on;
plot(ans.fbk_pos.Time, ans.fbk_pos.Data );
xlabel('Time [sec]');
ylabel('Position [rad]');
legend('command', 'feedback');
title('Position')
grid on

%%
rmpath('utils');
