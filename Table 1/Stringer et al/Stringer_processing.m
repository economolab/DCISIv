%%

path = 'E:\FDR Predictions DATA\Stringer et al\spks\';


animals = {'spksKrebs_Feb18.mat', 'spksRobbins_Feb18.mat', 'spksWaksman_Feb18'};

FRs = [];
ISI_viol = [];

for k = 1:3

    load(strcat(path, animals{k}))
    num_sessions = size(spks, 2);

    for i = 1:num_sessions
        
        
        st = spks(i).st;
        clu = spks(i).clu;
    
        spikes_by_neuron = cell(max(clu), 1);
    
        for j=1:length(st)
    
            spikes_by_neuron{clu(j)} = [spikes_by_neuron{clu(j)} st(j)];
    
        end
    
        for j=1:length(spikes_by_neuron)
    
            rec_time = max(spikes_by_neuron{j}) - min(spikes_by_neuron{j});
            FRs = [FRs; length(spikes_by_neuron{j})/rec_time];
            viols = sum(diff(sort(spikes_by_neuron{j})) < 0.0025);
            ISI_viol = [ISI_viol; viols/length(spikes_by_neuron{j})];
    
        end
    
    end
end

%% 

PSTHs = {};
ids = [];
id = 0;
for k = 1:3

    load(strcat(path, animals{k}))
    num_sessions = size(spks, 2);

    for i = 1:num_sessions
        
        id = id + 1;
        st = spks(i).st;
        clu = spks(i).clu;
        st = st - min(st);
        T = max(st);
    
        spikes_by_neuron = cell(max(clu), 1);
    
        for j=1:length(st)
    
            spikes_by_neuron{clu(j)} = [spikes_by_neuron{clu(j)} st(j)];
    
        end
    
        for j=1:length(spikes_by_neuron)
            
            unit = spikes_by_neuron{j};

            PSTHs{end+1} = genPSTH(unit, 1, T, round(T/0.1));
            ids = [ids; id];

        end
    
    end
end

%%

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




