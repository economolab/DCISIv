

%% asdf
myFolder = 'D:\FDR Predictions DATA\Inagaki et al 2019\SiliconProbeData\SiliconProbeData\FixedDelayTask\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

spikes_full = {};
trials_full = {};
ranges_full = [];
session_full = [];
unit_full = struct([]);
for i = 1:length(names)
    unit = load(strcat(myFolder,names{i})).unit;
    session_full = vertcat(session_full,repelem(i-1,length(unit))');
    [spikes,trials,ranges] = extract_times(unit);
    spikes_full = vertcat(spikes_full,spikes');
    trials_full = vertcat(trials_full,trials');
    ranges_full = vertcat(ranges_full,ranges);
    unit_full = [unit_full, unit];
end

writecell(spikes_full,'spikes.csv')
writecell(trials_full,'trials.csv')
writematrix(ranges_full,'ranges.csv')
writematrix(session_full,'sessions.csv')

num_bins = 120;
T = 6;
tau = 0.0025;

PSTHs = zeros(length(spikes_full), num_bins);

for i=1:length(spikes_full)
    num_trials = trials_full{i}(end) - trials_full{i}(1) + 1;
    PSTHs(i,:) = genPSTH(spikes_full{i}, num_trials, T, num_bins);
end

ISI_viol = zeros(length(spikes_full),1);

for i=1:length(spikes_full)
    viols = 0;
    num_spikes = 0;
    spikes = spikes_full{i};
    trials = trials_full{i};
    for j=ranges_full(i,1):ranges_full(i,2)
        trial_spikes = spikes(trials == j);
        viols = viols + sum(diff(trial_spikes) < tau);
        num_spikes = num_spikes + length(trial_spikes);
    end
    ISI_viol(i) = viols/num_spikes;
end

%% asdf
myFolder = 'D:\FDR Predictions DATA\Inagaki et al 2019\SiliconProbeData\SiliconProbeData\RandomDelayTask\withoutPerturbation\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

spikes_full = {};
trials_full = {};
ranges_full = [];
session_full = [];
unit_full = struct([]);
for i = 1:length(names)
    unit = load(strcat(myFolder,names{i})).unit;
    session_full = vertcat(session_full,repelem(i-1,length(unit))');
    [spikes,trials,ranges] = extract_times(unit);
    spikes_full = vertcat(spikes_full,spikes');
    trials_full = vertcat(trials_full,trials');
    ranges_full = vertcat(ranges_full,ranges);
    unit_full = [unit_full, unit];
end

writecell(spikes_full,'spikes.csv')
writecell(trials_full,'trials.csv')
writematrix(ranges_full,'ranges.csv')
writematrix(session_full,'sessions.csv')

num_bins = 80;
T = 4;
tau = 0.0025;

for i = 1:length(spikes_full)
    trials_full{i} = trials_full{i}(spikes_full{i} <= 4);
    spikes_full{i} = spikes_full{i}(spikes_full{i} <= 4);
end

PSTHs = zeros(length(spikes_full), num_bins);

for i=1:length(spikes_full)
    num_trials = trials_full{i}(end) - trials_full{i}(1) + 1;
    PSTHs(i,:) = genPSTH(spikes_full{i}, num_trials, T, num_bins);
end

ISI_viol = zeros(length(spikes_full),1);

for i=1:length(spikes_full)
    viols = 0;
    num_spikes = 0;
    spikes = spikes_full{i};
    trials = trials_full{i};
    for j=ranges_full(i,1):ranges_full(i,2)
        trial_spikes = spikes(trials == j);
        viols = viols + sum(diff(trial_spikes) < tau);
        num_spikes = num_spikes + length(trial_spikes);
    end
    ISI_viol(i) = viols/num_spikes;
end

%% asdf
myFolder = 'D:\FDR Predictions DATA\Inagaki et al 2019\SiliconProbeData\SiliconProbeData\RandomDelayTask\withPerturbation\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

spikes_full = {};
trials_full = {};
ranges_full = [];
session_full = [];
unit_full = struct([]);
for i = 1:length(names)
    unit = load(strcat(myFolder,names{i})).unit;
    session_full = vertcat(session_full,repelem(i-1,length(unit))');
    [spikes,trials,ranges] = extract_times(unit);
    spikes_full = vertcat(spikes_full,spikes');
    trials_full = vertcat(trials_full,trials');
    ranges_full = vertcat(ranges_full,ranges);
    unit_full = [unit_full, unit];
end

writecell(spikes_full,'spikes.csv')
writecell(trials_full,'trials.csv')
writematrix(ranges_full,'ranges.csv')
writematrix(session_full,'sessions.csv')

num_bins = 80;
T = 4;
tau = 0.0025;

for i = 1:length(spikes_full)
    trials_full{i} = trials_full{i}(spikes_full{i} <= 4);
    spikes_full{i} = spikes_full{i}(spikes_full{i} <= 4);
end

PSTHs = zeros(length(spikes_full), num_bins);

for i=1:length(spikes_full)
    num_trials = trials_full{i}(end) - trials_full{i}(1) + 1;
    PSTHs(i,:) = genPSTH(spikes_full{i}, num_trials, T, num_bins);
end

ISI_viol = zeros(length(spikes_full),1);

for i=1:length(spikes_full)
    viols = 0;
    num_spikes = 0;
    spikes = spikes_full{i};
    trials = trials_full{i};
    for j=ranges_full(i,1):ranges_full(i,2)
        trial_spikes = spikes(trials == j);
        viols = viols + sum(diff(trial_spikes) < tau);
        num_spikes = num_spikes + length(trial_spikes);
    end
    ISI_viol(i) = viols/num_spikes;
end



%% Functions

function [Rtot,F_v] = extract_Rtot_Fv(unit)

    SpikeTimes = {unit.SpikeTimes};
    Trial_idx_of_spike = {unit.Trial_idx_of_spike};
    Trial_info = {unit.Trial_info};

    Rtot = zeros(length(unit),1);
    F_v = zeros(length(unit),1);
    t_viol = 0.0025;
    
    for i = 1:length(unit)
    
        range = Trial_info{i}.Trial_range_to_analyze;
        num_trials = range(2) - range(1) + 1;
        T = num_trials*6; 
    
        Rtot(i) = length(SpikeTimes{i})/T;
    
        ISI_viols = 0;
        for j = range(1):range(2)
            indices = Trial_idx_of_spike{i} == j;
            trial = SpikeTimes{i}(indices);
            ISI_viols = ISI_viols + sum(diff(trial) < t_viol);
        end
    
        F_v(i) = ISI_viols/length(SpikeTimes{i});
    
    end
end

% extract trial range to analyze, spike times, and trial_idx for a given
% session
function [spikes, trials, ranges] = extract_times(unit)
    SpikeTimes = {unit.SpikeTimes};
    Trial_idx_of_spike = {unit.Trial_idx_of_spike};
    Trial_info = {unit.Trial_info};
    ranges = zeros(length(unit),2);
    spikes = SpikeTimes;
    trials = Trial_idx_of_spike;

    for i = 1:length(unit)
        range = Trial_info{i}.Trial_range_to_analyze;
        ranges(i,:) = range;
    end
    
end

function bins = genPSTH(spikes, n, T, N)
    delta = T/N;
    bins = zeros(N,1);
    for i=1:N
        for j=1:length(spikes)
            if (spikes(j) >= (i-1)*delta) && (spikes(j) < (i)*delta)
                bins(i) = bins(i) + 1;
            end
        end
    end 

    bins = bins/(delta*n);
end


% %% Recording times
% 
% times = zeros(length(ranges_full), 1);
% for i = 1:length(ranges_full)
%     range = ranges_full(i,:);
%     times(i) = range(end) - range(1) + 1;
% end
% 
% nans = [44, 144, 180, 311, 332, 388, 430, 554, 638, 640, 643, 699, 732];
% times_nan = zeros(length(nans), 1);
% for i = 1:length(times_nan)
%     range = ranges_full(nans(i),:);
%     times_nan(i) = range(end) - range(1) + 1;
% end
% 
% num_spikes = zeros(755, 1);
% for i = 1:755
%     num_spikes(i) = length(spikes_full{i});
% end
% 
% num_spikes_nan = zeros(length(nans), 1);
% for i = 1:length(num_spikes_nan)
%     num_spikes_nan(i) = length(spikes_full{nans(i)});
% end

% SpikeTimes = {unit.SpikeTimes};
% Trial_idx_of_spike = {unit.Trial_idx_of_spike};
% Trial_info = {unit.Trial_info};
% %%
% 
% Rtot = zeros(length(unit),1);
% F_v = zeros(length(unit),1);
% t_viol = 0.0025;
% 
% for i = 1:length(unit)
% 
%     range = Trial_info{i}.Trial_range_to_analyze;
%     num_trials = range(2) - range(1) + 1;
%     T = num_trials*6; 
% 
%     Rtot(i) = length(SpikeTimes{i})/T;
% 
%     ISI_viols = 0;
%     for j = range(1):range(2)
%         indices = Trial_idx_of_spike{i} == j;
%         trial = SpikeTimes{i}(indices);
%         ISI_viols = ISI_viols + sum(diff(trial) < t_viol);
%     end
% 
%     F_v(i) = ISI_viols/length(SpikeTimes{i});
% 
% end
% %%
% 
% times = [];
% for i = 1:length(unit)
%     times = vertcat(times,SpikeTimes{i});
% end
% 
% disp(min(times))
% disp(max(times))
% 
% %% asdf
% myFolder = 'C:\Users\jpv88\OneDrive\Documents\GitHub\Chand-Lab\SiliconProbeData\FixedDelayTask\';
% filePattern = fullfile(myFolder, '*.mat');
% theFiles = dir(filePattern);
% names = {theFiles.name};
% 
% Rtot_full = [];
% F_v_full = [];
% for i = 1:length(names)
%     unit = load(strcat(myFolder,names{i})).unit;
%     [Rtot,F_v] = extract_Rtot_Fv(unit);
%     Rtot_full = vertcat(Rtot_full,Rtot);
%     F_v_full = vertcat(F_v_full,F_v);
% end
% 
% writematrix(Rtot_full,'Rtot.csv')
% writematrix(F_v_full,'F_v.csv')


