function [] = transcript2evnt(subject,blocks,datatype,timing,elects,frqs)
%-Creates an structure, segmented by vocalization, containing data of your
% choosing
%-Can cycle through multiple blocks
%-Structure is hierarchical with phoneme, word, and phrase (e.g.
%word pair, sentence) levels. Levels correspond to tiers in the TextGrid
%file. Word and phoneme tiers are required. If no phrase tier exists,
%assumes that the word tier is equivalent to the vocalization tier.
%-Assumes usual organization to data (ie as it exists on cortex)
%-Call in subject folder
%-Calls ECoGGetEvents (internal) ECogReadData_Neural (internal)

%datatype: specifies the type of data to put into event structure.
%   0:Returns empty structure (just times, no data)
%   1:HG AA
%   2:Acoustics
%   3:Articualtor Kinematics
%   4:Raw ECoG
%   5:ANIN4
%   6:Formants
%   7:Beta AA
%   8:Theta AA
%   9:40 Band

%frqs: frequency bands to include (if applicable). Default is 8.

%timing: parameter defining timing before and after event to extract

%elects: electrodes to get data from (if applicable). Default is 256.

%formant: binary value dictating whether or not to extract formant values


%Example: transcript2evnt('EC6',[10,23,48],1,[0.7,0.8],1:8,1:64) will
%extract the HG AA for channels 1:64 and frq bands 1:8 for blocks 10,23,and
%48, with .7 and .8 seconds padding before and after each event

%Last Updated: 3/18/14 DFC

if ~exist('frqs','var') || isempty(frqs)
    frqs = 1:8;
end

if ~exist('timing','var') || isempty(timing)
    bef = 0.5; aft = 1; % in seconds
else
    bef = timing(1);aft = timing(2);
end
if ~exist('elects','var') || isempty(elects)
    elects = 1:256;
end
if ~exist('formant','var') || isempty(formant)
    formant =0;
end

for i = 1:length(blocks)
    ind = 0;
    bdir = [subject '_B' num2str(blocks(i))];
    
    %load the transcription file
    transcription = [subject '_B' num2str(blocks(i)) '_transcription_final.TextGrid'];
    if exist([bdir '/' subject '_B' num2str(blocks(i)) '_transcription_final.TextGrid'])
        transcription = [subject '_B' num2str(blocks(i)) '_transcription_final.TextGrid'];
        tmpevnt = ParseTextGrid([bdir '/' transcription]); %extract times of events and remove bad times
        
    elseif exist([bdir '/' subject '_B' num2str(blocks(i)) '_transcription_final.lab'])
        transcription = [subject '_B' num2str(blocks(i)) '_transcription_final.lab'];
        tmpevnt = ParseLab([bdir '/' transcription]); %extract times of events and remove bad times
        
    else
        error('Transcript File Not Found')
    end
    
    %load the bad channels file
    load([bdir '/Artifacts/badChannels.txt']);
    
    cnt1 = 1;
    disp([num2str(length(tmpevnt)) ' events found']);
    if datatype == 0
        NullEvnt = tmpevnt;
    end
    
    while cnt1<=length(tmpevnt)
        Start = 1000*(tmpevnt(cnt1).StartTime-bef); %
        Stop  = 1000*(tmpevnt(cnt1).StopTime+aft);
        if Start<1
            cnt1 = cnt1+1;
            continue;
            
        end
        %         disp(['event ' num2str(cnt1) ' out of ' num2str(length(tmpevnt))]);
        clc
        fprintf(['event ' num2str(cnt1) ' out of ' num2str(length(tmpevnt))]);
        if datatype == 1 %If HG AA
            newfs = 200;
            [AA_strct,fs,flag] = EvntReadData_Neural([bdir '/HilbAA_70to150_8band'],[Start Stop],elects,transcription,frqs,badChannels);
            if flag
                %down sample the data to newfs
                wszb = ceil(fs*.05); wsza = ceil(newfs*.05);
                for kk = 1:size(AA_strct,2)
                    temp2 = [zeros(length(elects),wszb) AA_strct(kk).elctrd zeros(length(elects),wszb)];
                    temp2 = resample(temp2',newfs*100,ceil(100*fs))';
                    AA_strct(kk).elctrd = temp2(:,wsza+1:end-wsza);
                end
                
                %%%%store information into structure
                ind = ind+1;
                Ref = -bef + tmpevnt(cnt1).StartTime;
                NeuralEvnt(ind).bef = bef;
                NeuralEvnt(ind).aft = aft;
                NeuralEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
                NeuralEvnt(ind).AbsStart  = Start;
                NeuralEvnt(ind).AbsStop   = Stop;
                NeuralEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
                NeuralEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
                NeuralEvnt(ind).Words     = tmpevnt(cnt1).Words;
                NeuralEvnt(ind).AA_data   = AA_strct;
                NeuralEvnt(ind).block     = blocks(i);
                NeuralEvnt(ind).fs        = newfs;
                NeuralEvnt(ind).frqs      = frqs;
                for j = 1:length(tmpevnt(cnt1).Words)
                    NeuralEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                    NeuralEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                    NeuralEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                    for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                        NeuralEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                        NeuralEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                        NeuralEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                    end
                end
            end
            NeuralEvnt(ind).spatialID = cellstr([repmat('electrode_',size(elects')) int2str(elects')]);
        elseif datatype == 2 %If Acoustics
            [Aud_strct,fs] = EvntReadData_Aud([bdir '/Analog'],[Start Stop]);
            
            ind = ind+1;
            
            Ref = -bef + tmpevnt(cnt1).StartTime;
            AudEvnt(ind).bef = bef;
            AudEvnt(ind).aft = aft;
            AudEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
            AudEvnt(ind).AbsStart  = Start;
            AudEvnt(ind).AbsStop   = Stop;
            AudEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
            AudEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
            AudEvnt(ind).Words     = tmpevnt(cnt1).Words;
            AudEvnt(ind).Aud_data   = Aud_strct;
            AudEvnt(ind).block     = blocks(i);
            AudEvnt(ind).fs        = fs;
            for j = 1:length(tmpevnt(cnt1).Words)
                AudEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                AudEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                AudEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                    AudEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                    AudEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                    AudEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                end
            end
            AudEvnt(ind).spatialID = 'acoustic_waveform';
        elseif datatype == 3 %If Kinematics
            [Kin_strct,fs,AlignInfo] = EvntReadData_Kin([bdir '/Kinematics'],[Start Stop]);
            
            ind = ind+1;
            
            Ref = -bef + tmpevnt(cnt1).StartTime;
            KinEvnt(ind).bef = bef;
            KinEvnt(ind).aft = aft;
            KinEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
            KinEvnt(ind).AbsStart  = Start;
            KinEvnt(ind).AbsStop   = Stop;
            KinEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
            KinEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
            KinEvnt(ind).Words     = tmpevnt(cnt1).Words;
            KinEvnt(ind).Kin_data  = Kin_strct;
            KinEvnt(ind).block     = blocks(i);
            KinEvnt(ind).fs        = fs;
            KinEvnt(ind).FTratio   = AlignInfo{1};
            KinEvnt(ind).USratio   = AlignInfo{2};
            KinEvnt(ind).lag       = AlignInfo{3};
            for j = 1:length(tmpevnt(cnt1).Words)
                KinEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                KinEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                KinEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                    KinEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                    KinEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                    KinEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                end
            end
            KinEvnt(ind).spatialID = [];
            if isfield(Kin_strct,'lips')
                KinEvnt(ind).spatialID = [KinEvnt(ind).spatialID; {'lips_upperBound','lips_lowerBound','lips_leftBound','lips_rightBound','jaw_y','jaw_x','nose_x','nose_y'}];
            end
            if isfield(Kin_strct,'tongue')
                KinEvnt(ind).spatialID = [KinEvnt(ind).spatialID; cellstr([repmat('tongue_x',100,1) int2str((1:100)')]);...
                    cellstr([repmat('tongue_y',100,1) int2str((1:100)')])];
            end
        elseif datatype == 4 %If Raw ECoG
            [Raw_strct,fs] = EvntReadData_Raw([bdir '/RawHTK'],[Start Stop],elects,badChannels);
            newfs = 400;
            wszb = ceil(fs*.05); wsza = ceil(newfs*.05);
            for kk = 1:size(Raw_strct,2)
                temp2 = [zeros(length(elects),wszb) Raw_strct(kk).elctrd zeros(length(elects),wszb)];
                temp2 = resample(temp2',newfs*100,ceil(100*fs))';
                raw_strct(kk).elctrd = temp2(:,wsza+1:end-wsza);
            end
            
            %%%%store information into structure
            ind = ind+1;
            Ref = -bef + tmpevnt(cnt1).StartTime;
            RawEvnt(ind).bef = bef;
            RawEvnt(ind).aft = aft;
            RawEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
            RawEvnt(ind).AbsStart  = Start;
            RawEvnt(ind).AbsStop   = Stop;
            RawEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
            RawEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
            RawEvnt(ind).Words     = tmpevnt(cnt1).Words;
            RawEvnt(ind).Raw_data   = Raw_strct;
            RawEvnt(ind).block     = blocks(i);
            RawEvnt(ind).fs        = newfs;
            RawEvnt(ind).frqs      = frqs;
            for j = 1:length(tmpevnt(cnt1).Words)
                RawEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                RawEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                RawEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                    RawEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                    RawEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                    RawEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                end
            end
        elseif datatype == 5 %If ANIN4
            [ANIN4_strct,fs] = EvntReadData_ANIN4([bdir '/Analog'],[Start Stop]);
            
            ind = ind+1;
            
            Ref = -bef + tmpevnt(cnt1).StartTime;
            ANIN4Evnt(ind).bef = bef;
            ANIN4Evnt(ind).aft = aft;
            ANIN4Evnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
            ANIN4Evnt(ind).AbsStart  = Start;
            ANIN4Evnt(ind).AbsStop   = Stop;
            ANIN4Evnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
            ANIN4Evnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
            ANIN4Evnt(ind).Words     = tmpevnt(cnt1).Words;
            ANIN4Evnt(ind).ANIN4_data   = ANIN4_strct;
            ANIN4Evnt(ind).block     = blocks(i);
            ANIN4Evnt(ind).fs        = fs;
            for j = 1:length(tmpevnt(cnt1).Words)
                ANIN4Evnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                ANIN4Evnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                ANIN4Evnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                    ANIN4Evnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                    ANIN4Evnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                    ANIN4Evnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                end
            end
        elseif datatype == 6 %If Formants
            [Form_strct,fs] = EvntReadData_Form([bdir '/Analog'],[Start Stop]);
            
            ind = ind+1;
            
            Ref = -bef + tmpevnt(cnt1).StartTime;
            FormEvnt(ind).bef = bef;
            FormEvnt(ind).aft = aft;
            FormEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
            FormEvnt(ind).AbsStart  = Start;
            FormEvnt(ind).AbsStop   = Stop;
            FormEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
            FormEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
            FormEvnt(ind).Words     = tmpevnt(cnt1).Words;
            FormEvnt(ind).Form_data   = Form_strct;
            FormEvnt(ind).block     = blocks(i);
            FormEvnt(ind).fs        = fs;
            for j = 1:length(tmpevnt(cnt1).Words)
                FormEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                FormEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                FormEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                    FormEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                    FormEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                    FormEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                end
            end
            FormEvnt(ind).spatialID = {'F0';'F1';'F2';'F3';'F4'};
        elseif datatype == 7 %If Beta AA
            newfs = 200;
            frqs = 14:21;
            [AA_strct,fs,flag] = EvntReadData_Neural([bdir '/HilbAA_4to200_40band'],[Start Stop],elects,transcription,frqs,badChannels);
            if flag
                %down sample the data to newfs
                wszb = ceil(fs*.05); wsza = ceil(newfs*.05);
                for kk = 1:size(AA_strct,2)
                    temp2 = [zeros(length(elects),wszb) AA_strct(kk).elctrd zeros(length(elects),wszb)];
                    temp2 = resample(temp2',newfs*100,ceil(100*fs))';
                    AA_strct(kk).elctrd = temp2(:,wsza+1:end-wsza);
                end
                
                %%%%store information into structure
                ind = ind+1;
                Ref = -bef + tmpevnt(cnt1).StartTime;
                BetaEvnt(ind).bef = bef;
                BetaEvnt(ind).aft = aft;
                BetaEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
                BetaEvnt(ind).AbsStart  = Start;
                BetaEvnt(ind).AbsStop   = Stop;
                BetaEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
                BetaEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
                BetaEvnt(ind).Words     = tmpevnt(cnt1).Words;
                BetaEvnt(ind).AA_data   = AA_strct;
                BetaEvnt(ind).block     = blocks(i);
                BetaEvnt(ind).fs        = newfs;
                BetaEvnt(ind).frqs      = frqs;
                for j = 1:length(tmpevnt(cnt1).Words)
                    BetaEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                    BetaEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                    BetaEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                    for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                        BetaEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                        BetaEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                        BetaEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                    end
                end
            end
            BetaEvnt(ind).spatialID = cellstr([repmat('electrode_',size(elects')) int2str(elects')]);
        elseif datatype == 8 %If Theta AA
            newfs = 200;
            frqs = 3:10;
            [AA_strct,fs,flag] = EvntReadData_Neural([bdir '/HilbAA_4to200_40band'],[Start Stop],elects,transcription,frqs,badChannels);
            if flag
                %down sample the data to newfs
                wszb = ceil(fs*.05); wsza = ceil(newfs*.05);
                for kk = 1:size(AA_strct,2)
                    temp2 = [zeros(length(elects),wszb) AA_strct(kk).elctrd zeros(length(elects),wszb)];
                    temp2 = resample(temp2',newfs*100,ceil(100*fs))';
                    AA_strct(kk).elctrd = temp2(:,wsza+1:end-wsza);
                end
                
                %%%%store information into structure
                ind = ind+1;
                Ref = -bef + tmpevnt(cnt1).StartTime;
                ThetaEvnt(ind).bef = bef;
                ThetaEvnt(ind).aft = aft;
                ThetaEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
                ThetaEvnt(ind).AbsStart  = Start;
                ThetaEvnt(ind).AbsStop   = Stop;
                ThetaEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
                ThetaEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
                ThetaEvnt(ind).Words     = tmpevnt(cnt1).Words;
                ThetaEvnt(ind).AA_data   = AA_strct;
                ThetaEvnt(ind).block     = blocks(i);
                ThetaEvnt(ind).fs        = newfs;
                ThetaEvnt(ind).frqs      = frqs;
                for j = 1:length(tmpevnt(cnt1).Words)
                    ThetaEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                    ThetaEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                    ThetaEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                    for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                        ThetaEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                        ThetaEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                        ThetaEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                    end
                end
            end
            ThetaEvnt(ind).spatialID = cellstr([repmat('electrode_',size(elects')) int2str(elects')]);
        elseif datatype == 9 %If 40 Band
            newfs = 200;
            [AA_strct,fs,flag] = EvntReadData_Neural([bdir '/HilbAA_4to200_40band'],[Start Stop],elects,transcription,frqs,badChannels);
            if flag
                %down sample the data to newfs
                wszb = ceil(fs*.05); wsza = ceil(newfs*.05);
                for kk = 1:size(AA_strct,2)
                    temp2 = [zeros(length(elects),wszb) AA_strct(kk).elctrd zeros(length(elects),wszb)];
                    temp2 = resample(temp2',newfs*100,ceil(100*fs))';
                    AA_strct(kk).elctrd = temp2(:,wsza+1:end-wsza);
                end
                
                %%%%store information into structure
                ind = ind+1;
                Ref = -bef + tmpevnt(cnt1).StartTime;
                FortyEvnt(ind).bef = bef;
                FortyEvnt(ind).aft = aft;
                FortyEvnt(ind).Phrase      = tmpevnt(cnt1).Phrase;
                FortyEvnt(ind).AbsStart  = Start;
                FortyEvnt(ind).AbsStop   = Stop;
                FortyEvnt(ind).StartTime = tmpevnt(cnt1).StartTime-Ref;
                FortyEvnt(ind).StopTime  = tmpevnt(cnt1).StopTime-Ref;
                FortyEvnt(ind).Words     = tmpevnt(cnt1).Words;
                FortyEvnt(ind).AA_data   = AA_strct;
                FortyEvnt(ind).block     = blocks(i);
                FortyEvnt(ind).fs        = newfs;
                FortyEvnt(ind).frqs      = frqs;
                for j = 1:length(tmpevnt(cnt1).Words)
                    FortyEvnt(ind).Words(j).Word = tmpevnt(cnt1).Words(j).Word;
                    FortyEvnt(ind).Words(j).StartTime = tmpevnt(cnt1).Words(j).StartTime-Ref;
                    FortyEvnt(ind).Words(j).StopTime = tmpevnt(cnt1).Words(j).StopTime-Ref;
                    for k = 1:length(tmpevnt(cnt1).Words(j).Phones)
                        FortyEvnt(ind).Words(j).Phones(k).Phon = tmpevnt(cnt1).Words(j).Phones(k).Phon;
                        FortyEvnt(ind).Words(j).Phones(k).StartTime = tmpevnt(cnt1).Words(j).Phones(k).StartTime - Ref;
                        FortyEvnt(ind).Words(j).Phones(k).StopTime = tmpevnt(cnt1).Words(j).Phones(k).StopTime - Ref;
                    end
                end
            end
            FortyEvnt(ind).spatialID = cellstr([repmat('electrode_',size(elects')) int2str(elects')]);
        end
        %         end
        cnt1 = cnt1+1;
    end
    
    
    %%%%%%%%%%%%%BASELINE STATISTICS%%%%%%%%%%%%%%%%%
    if datatype == 1
        if elects(1)>0 %get the mean and std for each electrode and channel
            
            bname = [bdir '/baseline.HG.mat'];
            exst = exist(bname,'file');
            sz = length(NeuralEvnt);
            if ~exst
                clc
                error('Baseline stats not found, run CalcBaseline.m')
                %                 [AA_strct,fs,flag] = EvntReadData_Neural([bdir '/HilbAA_70to150_8band'],-2,elects,transcription,frqs);
                %                 for kk=1:size(AA_strct,2)
                %                     NeuralEvnt(sz+1).AA_mu(:,kk) = AA_strct(kk).mu_std(:,1);
                %                     NeuralEvnt(sz+1).AA_std(:,kk) = AA_strct(kk).mu_std(:,2);
                %                 end
            else
                disp('Loading baseline statistics....');
                load(bname);  %load baseline data
                for kk=1:size(AA_strct,2)
                    NeuralEvnt(sz+1).AA_mu(:,kk) = baselineD(kk).mu_std(:,1);
                    NeuralEvnt(sz+1).AA_std(:,kk) = baselineD(kk).mu_std(:,2);
                end
                NeuralEvnt(sz+1).baselineMethod=method;
                NeuralEvnt(sz+1).baselineTime=baselineTime;
                NeuralEvnt(sz+1).baselineBlock=baselineBlock;
            end
        end
    elseif datatype == 7
        if elects(1)>0 %get the mean and std for each electrode and channel
            
            bname = [bdir '/baseline.beta.mat'];
            exst = exist(bname,'file');
            sz = length(BetaEvnt);
            if ~exst
                clc
                error('Baseline stats not found, run CalcBaseline.m')
            else
                disp('Loading baseline statistics....');
                load(bname);  %load baseline data
                for kk=1:size(AA_strct,2)
                    BetaEvnt(sz+1).AA_mu(:,kk) = baselineD(kk).mu_std(:,1);
                    BetaEvnt(sz+1).AA_std(:,kk) = baselineD(kk).mu_std(:,2);
                end
                BetaEvnt(sz+1).baselineMethod=method;
                BetaEvnt(sz+1).baselineTime=baselineTime;
                BetaEvnt(sz+1).baselineBlock=baselineBlock;
            end
        end
        
    elseif datatype == 8
        if elects(1)>0 %get the mean and std for each electrode and channel
            
            bname = [bdir '/baseline.theta.mat'];
            exst = exist(bname,'file');
            sz = length(ThetaEvnt);
            if ~exst
                clc
                error('Baseline stats not found, run CalcBaseline.m')
            else
                disp('Loading baseline statistics....');
                load(bname);  %load baseline data
                for kk=1:size(AA_strct,2)
                    ThetaEvnt(sz+1).AA_mu(:,kk) = baselineD(kk).mu_std(:,1);
                    ThetaEvnt(sz+1).AA_std(:,kk) = baselineD(kk).mu_std(:,2);
                end
                ThetaEvnt(sz+1).baselineMethod=method;
                ThetaEvnt(sz+1).baselineTime=baselineTime;
                ThetaEvnt(sz+1).baselineBlock=baselineBlock;
            end
        end
    elseif datatype == 9
        if elects(1)>0 %get the mean and std for each electrode and channel
            
            bname = [bdir '/baseline.forty.mat'];
            exst = exist(bname,'file');
            sz = length(FortyEvnt);
            if ~exst
                clc
                error('Baseline stats not found, run CalcBaseline.m')
            else
                disp('Loading baseline statistics....');
                load(bname);  %load baseline data
                for kk=1:size(AA_strct,2)
                    FortyEvnt(sz+1).AA_mu(:,kk) = baselineD(kk).mu_std(:,1);
                    FortyEvnt(sz+1).AA_std(:,kk) = baselineD(kk).mu_std(:,2);
                end
                FortyEvnt(sz+1).baselineMethod=method;
                FortyEvnt(sz+1).baselineTime=baselineTime;
                FortyEvnt(sz+1).baselineBlock=baselineBlock;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%SAVE RESULTS %%%%%%%%%%%%%%%%%%%%%%
    if datatype == 0
        save([bdir '/' subject '_B' num2str(blocks(i)) '_NullEvnt'],'NullEvnt','-v7.3');
    elseif datatype == 1
        save([bdir '/' subject '_B' num2str(blocks(i)) '_NeuralEvnt'],'NeuralEvnt','-v7.3');
    elseif datatype == 2
        save([bdir '/' subject '_B' num2str(blocks(i)) '_AudEvnt'],'AudEvnt','-v7.3');
    elseif datatype == 3
        save([bdir '/' subject '_B' num2str(blocks(i)) '_KinEvnt'],'KinEvnt','-v7.3');
    elseif datatype == 4
        save([bdir '/' subject '_B' num2str(blocks(i)) '_RawEvnt'],'RawEvnt','-v7.3');
    elseif datatype == 5
        save([bdir '/' subject '_B' num2str(blocks(i)) '_ANIN4Evnt'],'ANIN4Evnt','-v7.3');
    elseif datatype == 6
        save([bdir '/' subject '_B' num2str(blocks(i)) '_FormEvnt'],'FormEvnt','-v7.3');
    elseif datatype == 7
        save([bdir '/' subject '_B' num2str(blocks(i)) '_BetaEvnt'],'BetaEvnt','-v7.3');
    elseif datatype == 8
        save([bdir '/' subject '_B' num2str(blocks(i)) '_ThetaEvnt'],'ThetaEvnt','-v7.3');
    elseif datatype == 9
        save([bdir '/' subject '_B' num2str(blocks(i)) '_FortyEvnt' num2str(frqs)],'FortyEvnt','-v7.3');
    
    end
    clear NullEvnt NeuralEvnt KinEvnt AudEvnt RawEvnt FormEvnt BetaEvnt ThetaEvnt FortyEvnt
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [evnt,badsegments] = ParseTextGrid(TGfile)
%-extracts times of articulatory events from a TextGrid transcipt file
%-evnt structure is hierarchical with phoneme, word, and vocalization (e.g.
%word pair, sentence) levels. Levels correspond to tiers in the TextGrid
%file. Word and phoneme tiers are required. If no vocalization tier exists,
%assumes that the word tier is equivalent to the vocalization tier.
%-Called by ECogLoadEvents_* and ECogReadData_*
%-Removes events that coincide with bad time segments, if there are any

%Last Updated: Oct 14 2013 DFC

i = 1;
j = 1;
Pstarts = [];
Pends = [];
Phones = [];
Wstarts = [];
Wends = [];
Words = [];
PHstarts = [];
PHends = [];
Phrases = [];
badsegments = [];
fid = fopen(TGfile,'r');
tier = 1;
while ~feof(fid)
    tmp{i} = fgetl(fid);
    if findstr(tmp{i},'item [2]:')
        tier = 2;
    elseif findstr(tmp{i},'item [3]:')
        tier = 3;
    end
    if findstr(tmp{i},'text =')
        if strmatch(tmp{i}(21:end-2),'sp')
            continue;
        end
        if tier == 1     %If event is phon, record Phon times
            Phones = vertcat(Phones,{tmp{i}(21:end-2)});
            Pstarts = vertcat(Pstarts,str2num(tmp{i-2}(20:end)));
            Pends = vertcat(Pends,str2num(tmp{i-1}(20:end)));
        elseif tier == 2         %Else if event is word, record Word times
            Wstarts = vertcat(Wstarts,str2num(tmp{i-2}(20:end)));
            Wends = vertcat(Wends,str2num(tmp{i-1}(20:end)));
            Words = vertcat(Words,{tmp{i}(21:end-2)});
            
        elseif tier == 3    %Else if event is Phrase, record Phrase tiems
            Phrases = vertcat(Phrases,{tmp{i}(21:end-2)});
            PHstarts = vertcat(PHstarts,str2num(tmp{i-2}(20:end)));
            PHends = vertcat(PHends,str2num(tmp{i-1}(20:end)));
        end
    end
    i = i+1;
end

if tier == 3   %If Phrases Tier Exists
    %Find words that fall within Phrases.
    for PHind = 1:length(Phrases)
        evnt(PHind).Phrase = Phrases{PHind};
        evnt(PHind).StartTime = PHstarts(PHind);
        evnt(PHind).StopTime = PHends(PHind);
        
        startClose = find(abs((PHstarts(PHind) - Wstarts)) <= 0.01);  %1) the word starts near the start of the phrase
        startMore = find(Wstarts >= PHstarts(PHind));                 %2) the word starts after the start of the phrase
        stopClose = find(abs((PHends(PHind) - Wends)) <= 0.01);       %3) the word ends near the end of the phrase
        stopLess = find(Wends <= PHends(PHind));                      %4) the word ends before the end of the phrase
        
        %For the words that (1 or 2) are true AND (3 or 4) are true, those words are contained within the phrase
        startCandidates = unique([startClose;startMore]);
        stopCandidates  = unique([stopClose;stopLess]);
        ContainedWords = intersect(startCandidates,stopCandidates);
        for i = 1:length(ContainedWords)
            evnt(PHind).Words(i).Word = Words{ContainedWords(i)};
            evnt(PHind).Words(i).StartTime = Wstarts(ContainedWords(i));
            evnt(PHind).Words(i).StopTime = Wends(ContainedWords(i));
            
            %Find Phones that fall within Words
            startClose = find(abs((Wstarts(ContainedWords(i)) - Pstarts)) <= 0.01);
            startMore = find(Pstarts >= Wstarts(ContainedWords(i)));
            stopClose = find(abs((Wends(ContainedWords(i)) - Pends)) <= 0.01);
            stopLess = find(Pends <= Wends(ContainedWords(i)));
            
            startCandidates = unique([startClose;startMore]);
            stopCandidates  = unique([stopClose;stopLess]);
            ContainedPhones = intersect(startCandidates,stopCandidates);
            
            for j = 1:length(ContainedPhones)
                evnt(PHind).Words(i).Phones(j).Phon = Phones{ContainedPhones(j)};
                evnt(PHind).Words(i).Phones(j).StartTime = Pstarts(ContainedPhones(j));
                evnt(PHind).Words(i).Phones(j).StopTime = Pends(ContainedPhones(j));
            end
        end
    end
    
elseif tier == 2  %Else if only word tier exists, phrase tier = word tier
    for Wind = 1:length(Words)
        evnt(Wind).Phrase = Words{Wind};
        evnt(Wind).StartTime = Wstarts(Wind);
        evnt(Wind).StopTime = Wends(Wind);
        evnt(Wind).Words.Word = Words{Wind};
        evnt(Wind).Words.StartTime = Wstarts(Wind);
        evnt(Wind).Words.StopTime = Wends(Wind);
        
        %Find Phones that fall within Words
        startClose = find(abs((Wstarts(Wind) - Pstarts)) <= 0.01);
        startMore = find(Pstarts >= Wstarts(Wind));
        stopClose = find(abs((Wends(Wind) - Pends)) <= 0.01);
        stopLess = find(Pends <= Wends(Wind));
        
        startCandidates = unique([startClose;startMore]);
        stopCandidates  = unique([stopClose;stopLess]);
        ContainedPhones = intersect(startCandidates,stopCandidates);
        
        for j = 1:length(ContainedPhones)
            evnt(Wind).Words.Phones(j).Phon = Phones{ContainedPhones(j)};
            evnt(Wind).Words.Phones(j).StartTime = Pstarts(ContainedPhones(j));
            evnt(Wind).Words.Phones(j).StopTime = Pends(ContainedPhones(j));
        end
    end
    
end


fclose(fid);

% Check and exclude bad time segments
load([TGfile(1:find(double(TGfile)==47)) 'Artifacts/badTimeSegments.mat']);
if ~isempty(badTimeSegments)
    badind = 1;
    tol = .5; % in seconds
    blength = length(evnt);
    badsegments = badTimeSegments(:,[1,2]);
    % then exclude the segments that are in this file:
    for i=1:size(badTimeSegments,1)
        b = badsegments(i,1);
        e = badsegments(i,2);
        tmpevnt = evnt(1);
        tmpind  = 1;
        for cnt2 = 1:length(evnt)
            if ~((evnt(cnt2).StopTime>(b-tol)) && (evnt(cnt2).StopTime<(e+tol)))
                tmpevnt(tmpind) = evnt(cnt2);
                tmpind = tmpind+1;
            end
        end
        evnt = tmpevnt;
    end
    
    if (blength-length(evnt))>0,
        warning ([num2str(blength-length(evnt)) ' events were discarded...']);
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [evnt,badsegments] = ParseLab(labfile)
%extracts times of articulatory events from a transcipt file
%removes events that coincide with bad time segments, if there are any


fid = fopen(labfile,'r');
C = textscan(fid,'%s %f');
fclose(fid);

for i=2:2:length(C{1})
    CVs{i} = C{1}{i}(1:end-1);
    num = str2double(C{1}{i}(end));
    isplosive = ismember(CVs{i}(1:end-2),{'d','t','b','p','g','k','gh'});
    wordstart(i) = (~isplosive && num == 1 || (isplosive && num == 3));
    wordtrans(i) = (~isplosive && num == 3 || (isplosive && num == 4));
    wordstop(i)  = num == 2;
end

startinds = find(wordstart);
transinds = find(wordtrans);
stopinds  = find(wordstop);
Phrases = CVs(startinds);

%If the number of starts, transitions, and stops are not equal, then there
%is a typo within the transcript. Likely mislabeling of ending number.
if (length(startinds) ~= length(transinds)) || (length(transinds) ~= length(stopinds))
    minLength = min([length(startinds) length(transinds) length(stopinds)]);
    typos = [];
    typos = [typos find(diff(transinds(1:minLength) - startinds(1:minLength)))+1];
    typos = unique([typos find(diff(stopinds(1:minLength) - transinds(1:minLength)))+1]);
    typosTxt = sort([startinds(typos),transinds(typos),stopinds(typos)])/2;
    error(['Abnormal number of events detected. Typo exists within transcript. Suspected Lines of Lab: ' num2str(typosTxt)])
end

for i = 1:length(Phrases)
    evnt(i).Phrase = Phrases{i};
    evnt(i).StartTime = str2num(C{1}{startinds(i)+1})/1E7;
    %     evnt(i).StopTime = str2num(C{1}{stopinds(i)+1})/1E7;
    try
        evnt(i).StopTime = str2num(C{1}{stopinds(i)+1})/1E7;
    catch
        evnt(i).StopTime = C{2}(end-1)/1E7;
    end
    evnt(i).Words.Word = Phrases{i};
    evnt(i).Words.StartTime = evnt(i).StartTime;
    evnt(i).Words.StopTime = evnt(i).StopTime;
    
    %Consonant (C)
    evnt(i).Words(1).Phones(1).Phon = Phrases{i}(1:end-2);
    evnt(i).Words(1).Phones(1).StartTime = str2num(C{1}{startinds(i)+1})/1E7;
    evnt(i).Words(1).Phones(1).StopTime = str2num(C{1}{transinds(i)+1})/1E7;
    
    %Vowel (V)
    evnt(i).Words(1).Phones(2).Phon = Phrases{i}(end-1:end);
    evnt(i).Words(1).Phones(2).StartTime = str2num(C{1}{transinds(i)+1})/1E7;
    %     evnt(i).Words(1).Phones(2).StopTime = str2num(C{1}{stopinds(i)+1})/1E7;
    try
        evnt(i).Words(1).Phones(2).StopTime = str2num(C{1}{stopinds(i)+1})/1E7;
    catch
        evnt(i).Words(1).Phones(2).StopTime = C{2}(end-1)/1E7;
    end
end

% Check and exclude bad time segments
load([labfile(1:find(double(labfile)==47)) 'Artifacts/badTimeSegments.mat']);
if ~isempty(badTimeSegments)
    badind = 1;
    tol = .5; % in seconds
    blength = length(evnt);
    badsegments = badTimeSegments(:,[1,2]);
    % then exclude the segments that are in this file:
    for i=1:size(badTimeSegments,1)
        b = badsegments(i,1);
        e = badsegments(i,2);
        tmpevnt = evnt(1);
        tmpind  = 1;
        for cnt2 = 1:length(evnt)
            if ~((evnt(cnt2).StopTime>(b-tol)) && (evnt(cnt2).StopTime<(e+tol)))
                tmpevnt(tmpind) = evnt(cnt2);
                tmpind = tmpind+1;
            end
        end
        evnt = tmpevnt;
    end
    
    if (blength-length(evnt))>0,
        warning ([num2str(blength-length(evnt)) ' events were discarded...']);
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AA_strct,fs,flag] = EvntReadData_Neural(ddir,duration,elects,transcription,frqs,badchans)
%
%   elects: sequence of electrodes
%           -1: return the analog input (sound)
%   duration: start and stop times of data to extract
%               [0 0]: load all data
%               -2: return the mean and std
%   flag: if the duration of the data is the same as what was asked for, this
%           flag is 1, otherwise is zero.
%
if ~exist('transcription','var')
    transcription = [];
end

if ~exist('elects','var') || isempty(elects),
    elects = 1:256;
end

if ~exist('duration','var') || isempty(duration)
    duration = [];
end

AA_strct =[];

files = dir([ddir '/W*.htk']);
[d,fs,ch] = readhtk([ddir '/' files(1).name],[1 2]); %get sampling rate
clear('d');
if duration <0
    if ~isempty(transcription)
        tmp = find(double(transcription)==95);
        if strcmp(transcription(end-2:end),'lab')
            [evnt,badsegs] = ParseLab([transcription(1:tmp(2)-1) '/' transcription]); %for calculation of mean and STD
        else
            [evnt,badsegs] = ParseTextGrid([transcription(1:tmp(2)-1) '/' transcription]); %for calculation of mean and STD
        end
    else
        badsegs =[];
    end
end
flag = 0;
for cnt1 = 1:length(elects)
    temp = num2str(mod(elects(cnt1),64)); %recordings are organized into 4x64 chn blocks
    if strcmpi(temp,'0'), temp = '64';end
    fname = ['Wav' num2str(ceil(elects(cnt1)/64))  temp '.htk'];
    if duration(1)>=0 %% && duration(2)>=0 && length(duration)>1
        %extract and store the AA for this event
        if isempty(ddir) %calculate AA from Hilbert outputs
            ddirR = 'HilbReal_4to200_40band';
            dRl= readhtk([ddirR '/' fname],duration);
            ddirI = 'HilbImag_4to200_40band';
            dIm= readhtk([ddirI '/' fname],duration);
            data=abs(complex(dRl,dIm));
        else %use the High Gamma AA
            data = readhtk([ddir '/' fname],duration);
        end
        if ~isempty(find(badchans==cnt1)) %if its a bad channel, load it with NANs
            AA_strct(1).elctrd(cnt1,:) = nan(1,size(data,2));
            
        else
            if  length(frqs)==40
                AA_strct(1).elctrd(:,cnt1,:) = data;
            elseif size(data,1)>=length(frqs)
                AA_strct(1).elctrd(cnt1,:) = mean(data(frqs,:),1);
            else
                AA_strct(1).elctrd(cnt1,:) = nan(1,size(data,2));
            end
        end
    elseif length(duration) ==2 && find(duration <0)
        AA_strct(1).elctrd(cnt1,:) = nan(1);
    else
        duration
        AA_strct(1).electrd(cnt1,:) = nan(1,size(data,2));
    end
end
if length(duration) ==1 && duration == -2
    if ~exst
        save(bname,'baselineD');
    end
end
if length(duration)>1 && duration(1)>=0
    flag =1;
else
    flag = 0;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Aud_strct,fs] = EvntReadData_Aud(ddir,duration)

Aud_strct =[];

[tmp, fs] = wavread([ddir '/analog1.wav'],[1 2]);
samp_dur = floor(duration*(fs/1000));
[wav, fs] = wavread([ddir '/analog1.wav'],samp_dur);
wav = resample(wav,16000,fs);
fs = 10000;
Aud_strct.audio = wav';
Aud_strct.fs = fs;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Form_strct,fs] = EvntReadData_Form(ddir,duration)


Form_strct = [];

fs = 100;
file = dir([ddir '/*ifc_out.txt']);
formants = load([ddir '/' file.name]);
samp_dur2 = floor(duration*(100/1000));
Form_strct.formant = formants(samp_dur2(1):samp_dur2(2),:)';
Form_strct.fs = fs;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Kin_strct,fs,AlignInfo] = EvntReadData_Kin(ddir,duration,labfile)

if ~exist('duration','var') || isempty(duration)
    duration = [0 0];
end

if isempty(dir([ddir '/Art_data.mat']))
    Kinematic_align(ddir,0,0,0)
    load([ddir '/Art_data.mat']);
else
    load([ddir '/Art_data.mat']);
end

Kin_strct =[];
AlignInfo = cell(1,3);
duration_frames = floor(duration*(30/1000)); %Convert ms to frames
if ~duration
    Kin_strct.lips.x = Art_data.lips.x';
    Kin_strct.lips.y = Art_data.lips.y';
    Kin_strct.jaw.xy = Art_data.jaw.xy';
    Kin_strct.nose.xy = Art_data.nose.xy';
    if isfield(Art_data,'tongue')
        Kin_strct.tongue.x = Art_data.tongue.x';
        Kin_strct.tongue.y = Art_data.tongue.y';
    end
    fs = 30;
else
    if isfield(Art_data,'lips')
        Kin_strct.lips.x = Art_data.lips.x(duration_frames(1):duration_frames(2),:)';
        Kin_strct.lips.y = Art_data.lips.y(duration_frames(1):duration_frames(2),:)';
        Kin_strct.jaw.xy = Art_data.jaw.xy(duration_frames(1):duration_frames(2),:)';
        Kin_strct.nose.xy = Art_data.nose.xy(duration_frames(1):duration_frames(2),:)';
        AlignInfo{1} = Art_data.FTratio;
    end
    if isfield(Art_data,'tongue')
        Kin_strct.tongue.x = Art_data.tongue.x(duration_frames(1):duration_frames(2),:)';
        Kin_strct.tongue.y = Art_data.tongue.y(duration_frames(1):duration_frames(2),:)';
        AlignInfo{2} = Art_data.USratio;
        AlignInfo{3} = Art_data.lag;
    end
    fs = 30;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Raw_strct,fs] = EvntReadData_Raw(ddir,duration,elects,badChannels)
if ~exist('transcription','var')
    transcription = [];
end

if ~exist('elects','var') || isempty(elects),
    elects = 1:256;
end

if ~exist('duration','var') || isempty(duration)
    duration = [];
end

Raw_strct =[];

files = dir([ddir '/W*.htk']);
[d,fs,ch] = readhtk([ddir '/' files(1).name],[1 2]); %get sampling rate
clear('d');
for cnt1 = 1:length(elects)
    temp = num2str(mod(elects(cnt1),64)); %recordings are organized into 4x64 chn blocks
    if strcmpi(temp,'0'), temp = '64';end
    fname = ['Wav' num2str(ceil(elects(cnt1)/64))  temp '.htk'];
    data = readhtk([ddir '/' fname],duration);
    if ~isempty(find(badChannels==cnt1)) %if its a bad channel, load it with NANs
        Raw_strct(1).elctrd(cnt1,:) = nan(1,size(data,2));
    else
        Raw_strct(1).elctrd(cnt1,:) = data;
    end
end
end

function [ANIN4_strct,fs] = EvntReadData_ANIN4(ddir,duration)

ANIN4_strct =[];

[tmp, fs] = wavread([ddir '/ANIN4.wav'],[1 2]);
samp_dur = floor(duration*(fs/1000));
[wav, fs] = wavread([ddir '/analog1.wav'],samp_dur);
% wav = resample(wav,16000,fs);
% fs = 10000;
ANIN4_strct.data = wav';
ANIN4_strct.fs = fs;
end