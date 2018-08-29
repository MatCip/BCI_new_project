

%% 




[~, directory_name] = uigetfile('*.mat');
fs=500;

file_list=dir(directory_name);


for i=1:length(file_list)
    if(strcmp(file_list(i).name,'.')==0 && strcmp(file_list(i).name,'..')==0 && strcmp(file_list(i).name,'.DS_Store')==0)
        data = load(fullfile(directory_name, file_list(i).name));
        disp(file_list(i).name);
        [labels,signal]=extract_labels(data.eeg);
        highfiltered=data.eeg.highpassFiltered;
        if(isempty(labels))
            disp('>>> It is empty');
        end
        name_file=['Signal_label_',file_list(i).name];
        
        save(['Autoencoder_signal_',file_list(i).name],'highfiltered');
        save(name_file,'signal','labels');
    end
    
end

function [labels,signal]=extract_labels(eeg)
    
     yes_start=find(eeg.onset==4);
     no_start=find(eeg.onset==8);
     no_end=find(eeg.onset==2);
     yes_end=find(eeg.onset==1);
     labels=ones((length(yes_start)+length(no_start)),1);
     i=1;
     j=1;
     k=1;
     cont=0;
     
     while(k<=(length(yes_start)+length(no_start)))
         cont=cont+1;
         if(j>length(yes_start))
            temp_signal=eeg.highpassFiltered(:,no_start(i):no_end(i));
            if(isempty(temp_signal)==0)
            signal(k,:,:)=temp_signal(:,1:2501);

            labels(k)=0;
             k=k+1;
            
             break
            end
         end
         
         if(i>length(no_start))
             temp_signal=eeg.highpassFiltered(:,yes_start(j):yes_end(j));
             if(isempty(temp_signal)==0)
             signal(k,:,:)=temp_signal(:,1:2501);
              k=k+1;
             break
             end
         end
         
         if(no_start(i)<yes_start(j))
             
            temp_signal=eeg.highpassFiltered(:,no_start(i):no_end(i));
            if(isempty(temp_signal)==0)
            signal(k,:,:)=temp_signal(:,1:2501);
            labels(k)=0;
            i=i+1;
             k=k+1;
            end
            
         else
            
            temp_signal=eeg.highpassFiltered(:,yes_start(j):yes_end(j));
            if(isempty(temp_signal)==0)
            signal(k,:,:)=temp_signal(:,1:2501);
            j=j+1;
            k=k+1;
            end
         end
         
        if(cont>100)
            
            disp('ciao')
        end

         
     end
     
   
  
     
     
    
%      yes_end=find(eeg.onset==1);
%      
%      filtered=transpose(eeg.highpassFiltered);
%         for k=1:length(yes_start)
%             yes_signal{k}=filtered(yes_start(k):yes_end(k),:);
%         end
%         no_start=find(eeg.onset==8);
%         no_end=find(eeg.onset==2);
%         for k=1:length(no_start)
%             no_signal{k}=filtered(no_start(k):no_end(k),:);
%         end
%         EEGyes=yes_signal;
%         EEGno=no_signal;
%         
        
        
        
        
        
end
        
        
       
        