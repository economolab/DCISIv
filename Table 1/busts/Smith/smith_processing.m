num_units = size(ex.EVENTS, 1);
num_stims = size(ex.EVENTS, 2);
spikes = cell(num_units, 1);

for i = 1:num_units
    
    for j = 1:num_stims

        stim = ex.EVENTS(i, j, 1:ex.REPEATS(j));
        stim = squeeze(stim);
        spikes{i} = [spikes{i}; stim];

    end

end

 %%

num_spikes = 0;
num_viols = 0;
ISI_viol = [];
ISIs = [];
PSTHs = zeros(num_units, 40);

for i = 1:num_units
    
    num_spikes = 0;
    num_viols = 0;
    trials = spikes{i};

    for j = 1:length(trials)

        num_spikes = num_spikes + length(trials{j});
        num_viols = num_viols + sum(diff(trials{j}) < 0.002);
        ISIs = [ISIs; diff(trials{j})];

    end

    ISI_viol = [ISI_viol; num_viols/num_spikes];
    unit_spikes = cell2mat(trials);

    unit_spikes = unit_spikes + 0.5;

    if (min(unit_spikes) <= 0) && (max(unit_spikes) >= 2)

        PSTHs(i,:) = genPSTH(unit_spikes, length(trials), 2, 40);

    else

        disp('error')
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
