close all; clear; clc;
%% Simulink�ɦW
model_name = 'Two_mass_sim.slx';
addpath('utils');
%% �����I�ҫ��ѼƳ]�w
J1 = 0.195e-3; % ���F�ݺD�q

freq_r = 550; % �@���W�v(Hz)
freq_ar = 500; % �Ϧ@���W�v(Hz)
zeta = 0.001; % �����Y��

omega_r = 2 * pi * freq_r; % �@���W�v(Hz)
omega_ar = 2 * pi * freq_ar; % �Ϧ@���W�v(Hz)

%% ��L�]�w
% Command Input --- �z�LS curve�ͦ���m�B�t�סB�[�t�שR�O
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

% Parameter Setting --- �]�w����Ѽ�
BW = 100;
KPP = BW * 0.5 * pi;
KVP = BW * 2 * pi;
Jm = J1;

%% �p������q�ҫ���L�Ѽ� J2�BC12�BK12
J2 = (omega_r^2*J1/omega_ar^2) - J1;
meff = (J1 * J2) / (J1 + J2); % ���Ľ�q

K12 = omega_ar^2 * J2;
C12 = zeta*(2*sqrt(meff*K12));

% �ಾ��ƫغc (²������)
s = tf('s');
system = ((J2 * s^2) + K12 + C12 * s) / (J1 * J2 * s^4 + (J1 + J2) * C12 * s^3 + (J1 + J2) * K12 * s^2);

% ����ಾ���
disp('�ಾ��� G(s):');
system

% �@���W�v�P�Ϧ@���W�v
omega_r_ = sqrt(K12 / meff);
omega_ar_ = sqrt(K12 / J2);

% �����Y��
zeta_ = C12 / (2 * sqrt(meff * K12));

% ��ܭp�⵲�G
fprintf('�@���W�v (Hz): %.4f\n', omega_r_ / (2 * pi));
fprintf('�Ϧ@���W�v (Hz): %.4f\n', omega_ar_ / (2 * pi));
fprintf('�����Y��: %.4f\n', zeta_);

% Bode ���˵�
figure;
bode(system);
grid on;
title('�����I�t�� Bode ��');

%% bode plot by Simulink
io = getlinio('Two_mass_sim');
sys_simulink = linearize('Two_mass_sim', [io(1), io(2)]);
% [Gm_velocity, Pm_velocity, ~, ~] = margin(sys_velocity);
% Gm_velocity = 20 * log10(Gm_velocity);
frequency_bode = 10.^(3:0.01:4);
figure
margin(sys_simulink, frequency_bode)

%% �������G
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
