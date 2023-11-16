myDir = 'E:\FDR Predictions DATA\Cheng et al\rawSpikeTime';
files = dir(myDir);


FRs = [];
ISI_viol = [];

for i = 1:length(files)

    if files(i).isdir == 0

        load(strcat(myDir, '/', files(i).name))
        unit_ids = unique(cluster_class(:,1));
        
        for j = 1:length(unit_ids)

            unit_id = unit_ids(j);
            spks = cluster_class(cluster_class(:,1) == unit_id,2);
            rec_t = max(spks) - min(spks);
            FRs = [FRs; length(spks)/rec_t];
            


        end

    end

end
