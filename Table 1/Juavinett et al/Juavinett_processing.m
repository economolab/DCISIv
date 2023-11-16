
myFolder = 'E:\FDR Predictions DATA\Juavinett et al\Neuropixels Collaboration\concatenated\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

for m = 1:length(names)
    load(strcat(myFolder, names{m}))

    % 0-indexed 
    cluster_ids = sp.clu;
    spike_times = sp.st;
    
    num_clu = max(cluster_ids)+1;
    
    spikes = cell(num_clu,1);
    
    for i = 1:length(spike_times)
        disp(i)
        spikes{cluster_ids(i)+1} = [spikes{cluster_ids(i)+1} spike_times(i)];
    end
    
    spikes = spikes(~cellfun('isempty',spikes));
    
    T = 10;
    tau = 0.0025;
    
    N_cells = length(spikes);
    PSTHs = zeros(N_cells, 200);
    ISI_viol = zeros(1, N_cells);
    
    for i = 1:length(spikes)
    
        unit_times = spikes{i}';
    
        t_start = unit_times(1);
        t_stop = unit_times(end);
    
        t_start = ceil(t_start/10)*10;
        t_stop = floor(t_stop/10)*10;
        
        unit_spikes = unit_times((unit_times >= t_start) & (unit_times <= t_stop));
        unit_spikes = unique(unit_spikes);
    
        viols = sum(diff(unit_spikes) < tau);
        total_spikes = length(unit_spikes);
        ISI_viol(i) = viols/total_spikes;
    
        n_trials = (t_stop - t_start)/T;
    
        start_trial = t_start/T;
        end_trial = start_trial + n_trials;
        
        spikes_aligned = [];
        for j = start_trial:end_trial-1
            spikes_subset = unit_spikes((unit_spikes >= j*T) & (unit_spikes < (j+1)*T));
            spikes_aligned = [spikes_aligned; spikes_subset-j*T];
        end
        
        PSTHs(i,:) = genPSTH(spikes_aligned, n_trials, T, 200);
    
    end

    save(strcat(names{m}, '_PSTHs.mat'), "PSTHs")
    save(strcat(names{m}, '_ISIviol.mat'), 'ISI_viol')
end

%% obtain homogeneous firing rates

myFolder = 'E:\FDR Predictions DATA\Juavinett et al\Neuropixels Collaboration\concatenated\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

FRs = {};
ISI_viol = {};

for m = 1:length(names)

    disp(m)
    load(strcat(myFolder, names{m}))

    % 0-indexed 
    cluster_ids = sp.clu;
    spike_times = sp.st;

    [cluster_ids,I] = sort(cluster_ids);
    spike_times = spike_times(I);

    switch_points = find(diff(cluster_ids));

    num_clu = length(switch_points) + 1;
    
    spikes = cell(num_clu,1);

    for k = 1:length(spikes)

        if k == 1

            spikes{k} = spike_times(1 : switch_points(1));
            continue

        elseif k == length(spikes)

            spikes{k} = spike_times(switch_points(end) + 1 : end);
            continue

        end

        spikes{k} = spike_times(switch_points(k-1) + 1 : switch_points(k));

    end

    for k = 1:length(spikes)
        
        num_spikes = length(spikes{k});
        rec_time = max(spikes{k}) - min(spikes{k});
        FRs{end+1} = num_spikes/rec_time;
        viols = sum(diff(spikes{k}) < 0.0025);
        ISI_viol{end+1} = viols/num_spikes;
        
    end

end


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