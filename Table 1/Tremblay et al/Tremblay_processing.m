ISIs = [];

for i = 1:length(SpikeData)
    electrode = SpikeData{i};
    
    for j = 1:length(electrode)
        spikes = electrode{j};
        ISIs = [ISIs; diff(spikes)];
    end
    
end
