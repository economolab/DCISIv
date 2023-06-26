load('D:\FDR Predictions DATA\Mallory et al\Freely_moving_data_with_inertial_sensor.mat')

%% PSTH per cell, 50 ms bins

T = 10;
tau = 0.0025;

N_cells = length(cell_info);
PSTHs = zeros(N_cells, 200);
ISI_viol = zeros(1, N_cells);

for i = 1:177

    spike_times = cell_info(i).spike_times;

    t_start = spike_times(1);
    t_stop = spike_times(end);

    t_start = ceil(t_start/10)*10;
    t_stop = floor(t_stop/10)*10;
    
    spikes = spike_times((spike_times >= t_start) & (spike_times <= t_stop));
    spikes = unique(spikes);

    viols = sum(diff(spikes) < tau);
    total_spikes = length(spikes);
    ISI_viol(i) = viols/total_spikes;

    n_trials = (t_stop - t_start)/T;

    start_trial = t_start/T;
    end_trial = start_trial + n_trials;
    
    spikes_aligned = [];
    for j = start_trial:end_trial-1
        spikes_subset = spikes((spikes >= j*T) & (spikes < (j+1)*T));
        spikes_aligned = [spikes_aligned; spikes_subset-j*T];
    end
    
    PSTHs(i,:) = genPSTH(spikes_aligned, n_trials, T, 200);

end

%%

save('mallory_PSTHs.mat', "PSTHs")
save('mallory_ISI_viol.mat', 'ISI_viol')

%%

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

% %%
% 
% animal_id = string({cell_info.animal_id});
% time = {cell_info.time};
% uniq_ids = unique(animal_id);
% 
% for i = 1:length(uniq_ids)
%     mask = (animal_id == uniq_ids(i));
%     cell_time = time(mask);
% 
% end
% 
% %%
% 
% shortest_time = 10000000000;
% for i = 1:length(time)
%     if length(time{i}) < shortest_time
%         shortest_time = length(time{i});
%     end
% end
