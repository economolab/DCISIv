myFolder = 'C:\Users\jpv88\Documents\FDR Predictions DATA\Economo et al\ProcessedData\';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
names = {theFiles.name};

tau = 0.0025;
ISI_viol = [];
PSTHs = [];
T = 6;

just_Tagged = false;

for m = 1:length(names)
    load(strcat(myFolder, names{m}))
    sessions = length(obj);
    for i = 1:sessions
        neurons = length(obj{i}.eventSeriesHash.value);
        for j = 1:neurons
            
            hash = obj{i}.eventSeriesHash.value(j);
            if (just_Tagged == true) && (hash{1}.collision.pass == 0) 
                continue
            end
            eventTimes = hash{1}.eventTimes;
            eventTrials = hash{1}.eventTrials;
            eventTrials = eventTrials(eventTimes >= -3 & eventTimes <= 3);
            eventTimes = eventTimes(eventTimes >= -3 & eventTimes <= 3);
    
            min_trial = min(eventTrials);
            max_trial = max(eventTrials);
    
            num_spikes = 0;
            viols = 0;
            
            % ISI_hist = [];
            for k = min_trial:max_trial
                mask = (eventTrials == k);
                spikes = eventTimes(mask);
                num_spikes = num_spikes + length(spikes);
                viols = viols + sum(diff(spikes) < tau);
                % ISI_hist = [ISI_hist diff(spikes)'];
            end

            if ~isempty(T) && (length(eventTimes) > 2)
                ISI_viol = [ISI_viol; viols/num_spikes];
        
                trials = max_trial - min_trial + 1;
                
                eventTimes = eventTimes + 3;
                PSTH = genPSTH(eventTimes, trials, T, 120);
                PSTHs = [PSTHs; PSTH'];
            end
    
        end
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