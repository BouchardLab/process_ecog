function [] = evnt2Ymatrix(subject,blocks,datatype,task,t_zero,timing,repeat,stim_type,frqs)

%datatype: specifies the type of data is in the evnt structure.
%   1:HG AA
%   2:Acoustics
%   3:Articualtor Kinematics
%   4:Raw ECoG
%   5:EGG
%   6:Formants
%   7:Beta AA
%   8:Theta AA
%   9:Forty Band
%
%task: specifies the task and thus what labels to include in the matrix.
%   1:vowels in isolation
%   2:hVd
%   3:BDG
%   4:CV_td_CV
%   5:CV
%   6:Coarticulation 1 word
%   7:Coarticulation 2 word
%
%t_zero: specifies what word and phoneme transition to align to, starting with
%onset. E.g. t_zero = {3 2} will align to the transition between the 1st and 2nd
%phoneme of the 3rd word. Can also specify by end of word, e.g. {2 -2} 
%will give you the second to last transition of the second word. 
%Defaults to beginning of first word. 
%Requires that each phrase has the same number of words
%Be sure that evnt structure you pull from has appropriately long bef 
%and aft times to acoomodate.
%
%timing: specifies how much time to grab before the onset of the phoneme of
%choice, and how much after the trial end. e.g. [0.75 1] will give you 750
%ms before and 1 sec after
%
%repeat: specifies whether this was a repetition task or not
%
%stim_type: production (2) or perception (1) (in a repetition task)
%
%Last Updated: 3/10/15 DFC


if ~exist('t_zero','var') || isempty(t_zero)
    t_zero = {1 1};
end

if ~exist('timing','var') || isempty(timing)
    n_bef = 0.5; n_aft = 0.8; timing = [0.5 0.8];
else
    n_bef = timing(1);  n_aft = timing(2);
end

if ~exist('repeat','var') || isempty(repeat)
    repeat = 1;
end

if ~exist('stim_type','var') || isempty(stim_type)
    stim_type = 1;
end

stim_type_id = {'perc','prod'};

%Set Task
if task == 1 %vowels
    wlist = {'AAA','AEE','AHH','EHH','ERR','IHH','IYY','UHH','UWW'};
    tname = 'vow';
elseif task == 2  %hVds
    wlist = {'HAD','HAWED','HEAD','HEARD','HEED','HID','HOOD','HUD','WHOOD'};
    tname = 'hVd';
elseif task == 3  %BDG
    wlist = {'BOO','DOO','GOO','BAA','DAA','GAA'};
    tname = 'BDG';
elseif task == 4  %CV_td_CV
    wlist = {'LEE_SEE','LEE_KEE','LEE_SOO','LEE_KOO'};
    tname = 'CVtdCV';
elseif task == 5 %CVs
    wlist = {'baa','bee','boo','daa','dee','doo','faa','fee','foo','gaa','gee','goo','haa','hee','hoo',...
        'kaa','kee','koo','laa','lee','loo','maa','mee','moo','naa','nee','noo','paa','pee','poo','raa','ree','roo',...
        'saa','shaa','shee','shoo','see','soo','taa','thaa','thee','thoo','tee','too','vaa','vee','voo','waa','wee','woo',...
        'yaa','yee','yoo','zaa','zee','zoo'};
    wlist = upper(wlist);
    tname = 'CV';
elseif task == 6 %Coart1
    wlist = {'SKEET','LEAK','SEAT','LEAKED','SCOOT','SUIT','LEAKST','LEE'};
    wlist = upper(wlist);
    tname = 'Coart1';
elseif task == 7 %Coart2
    wlist = {'LEE SEAT','LEE SKEET','LEAK SKEET','LEAKED SKEET','LEAKST SKEET',...
        'LEE SUIT','LEE SCOOT','LEAK SCOOT','LEAKED SCOOT','LEAKST SCOOT'};
    wlist = upper(wlist);
    tname = 'Coart2';
end


for i = 1:length(blocks)
    clear EvntSampLength SampLength Start Stop StartSamp StopSamp
    
    disp(['block ' num2str(blocks(i))])
    %Load Data
    if datatype == 1
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_NeuralEvnt.mat'])
        Struct = NeuralEvnt;
        prefix = {'Neural'};
        suffix = {'AA_data.elctrd'};
        numFeatures = size(NeuralEvnt(1).AA_data.elctrd,1);
        baselineMethod{i} = NeuralEvnt(end).baselineMethod;
        baselineTime{i} = NeuralEvnt(end).baselineTime;
        baselineBlock{i} = NeuralEvnt(end).baselineBlock;
        baselineMu{i}{1} = NeuralEvnt(end).AA_mu;
        baselineSTD{i}{1} = NeuralEvnt(end).AA_std;
    elseif datatype == 2
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_AudEvnt.mat'])
        Struct = AudEvnt;
        prefix = {'Aud'};
        suffix = {'Aud_data.audio'};
        formant = 0;
        numFeatures = 1;
    elseif datatype == 3
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_KinEvnt.mat'])
        Struct = KinEvnt;
        prefix = {'Kin'};
        if isfield(KinEvnt(1).Kin_data,'tongue')
            FT_only = 0;
            suffix = {'Kin_data.tongue.x'};
            numFeatures = 208;
            if ~isfield(KinEvnt(1).Kin_data,'lips')
                US_only = 1;
                numFeatures = 200;
            end
        else
            FT_only = 1;
            suffix = {'Kin_data.lips.x'};
            numFeatures = 8;
        end
    elseif datatype == 4
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_RawEvnt.mat'])
        Struct = RawEvnt;
        prefix = {'Raw'};
        suffix = {'Raw_data.elctrd'};
        numFeatures = size(RawEvnt(1).Raw_data.elctrd,1);
    elseif datatype == 5
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_ANIN4Evnt.mat'])
        Struct = ANIN4Evnt;
        prefix = {'ANIN4'};
        suffix = {'ANIN4_data.data'};
        numFeatures = size(ANIN4Evnt(1).ANIN4_data,1);
    elseif datatype == 6
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_FormEvnt.mat'])
        Struct = FormEvnt;
        prefix = {'Form'};
        suffix = {'Form_data.formant'};
        formant = 1;
        numFeatures = 5;
    elseif datatype == 7
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_BetaEvnt.mat'])
        Struct = BetaEvnt;
        prefix = {'Beta'};
        suffix = {'AA_data.elctrd'};
        numFeatures = size(BetaEvnt(1).AA_data.elctrd,1);
        baselineMethod{i} = BetaEvnt(end).baselineMethod;
        baselineTime{i} = BetaEvnt(end).baselineTime;
        baselineBlock{i} = BetaEvnt(end).baselineBlock;
        baselineMu{i}{1} = BetaEvnt(end).AA_mu;
        baselineSTD{i}{1} = BetaEvnt(end).AA_std;
    elseif datatype == 8
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_ThetaEvnt.mat'])
        Struct = ThetaEvnt;
        prefix = {'Theta'};
        suffix = {'AA_data.elctrd'};
        numFeatures = size(ThetaEvnt(1).AA_data.elctrd,1);
        baselineMethod{i} = ThetaEvnt(end).baselineMethod;
        baselineTime{i} = ThetaEvnt(end).baselineTime;
        baselineBlock{i} = ThetaEvnt(end).baselineBlock;
        baselineMu{i}{1} = ThetaEvnt(end).AA_mu;
        baselineSTD{i}{1} = ThetaEvnt(end).AA_std;
    elseif datatype == 9
        load([subject '_B' num2str(blocks(i)) '/' subject '_B' num2str(blocks(i)) '_FortyEvnt' num2str(frqs) '.mat'])
        Struct = FortyEvnt;
        prefix = {['Forty' num2str(frqs)]};
        suffix = {'AA_data.elctrd'};
        numFeatures = size(FortyEvnt(1).AA_data.elctrd,1);
        baselineMethod{i} = FortyEvnt(end).baselineMethod;
        baselineTime{i} = FortyEvnt(end).baselineTime;
        baselineBlock{i} = FortyEvnt(end).baselineBlock;
        baselineMu{i}{1} = FortyEvnt(end).AA_mu;
        baselineSTD{i}{1} = FortyEvnt(end).AA_std;
    end
    
    bef = Struct.bef;
    aft = Struct.aft;
    fs = Struct.fs;
    
    %Clean up transcriptions, pick out events of interest
    include =[]; exclude = [];
    for j = 1:length(Struct)
        %Note baseline stats
        if (datatype == 1 || datatype == 7 || datatype == 8 || datatype == 9) && j == length(Struct)
            baseline{i} = [Struct(j).AA_mu Struct(j).AA_std];
            continue;
        end
        
        
        
        %Fix capitalization
        Struct(j).Phrase = upper(Struct(j).Phrase);
        for k = 1:length(Struct(j).Words)
            Struct(j).Words(k).Word = upper(Struct(j).Words(k).Word);
            for kk = 1:length(Struct(j).Words(k).Phones)
                Struct(j).Words(k).Phones(kk).Phon = upper(Struct(j).Words(k).Phones(kk).Phon);
            end
        end
        
        %Fix common typos for CVs
        if task == 5 && ~repeat
            if strcmp(Struct(j).Phrase(end-1:end),'UW') || strcmp(Struct(j).Phrase(end-1:end),'UU') || strcmp(Struct(j).Phrase(end-1:end),'HO') || strcmp(Struct(j).Phrase(end-1:end),'UE')
                Struct(j).Phrase(end-1:end) = 'OO';
            elseif strcmp(Struct(j).Phrase(end-1:end),'AW')
                Struct(j).Phrase(end-1:end) = 'AA';
            elseif strcmp(Struct(j).Phrase(1:2),'GH')
                Struct(j).Phrase(2) = [];
            end
        end
        
        %Select only the production events and events specified in input words
        
        if repeat
            %             disp([Struct(j).Phrase num2str(j)]);
            if stim_type == 2
                if strcmp(Struct(j).Phrase(end),'2') && ismember(Struct(j).Phrase(1:end-1),wlist)
                    include = [include j];
                    
                    %Remove Tailing number from label if repetition
                    Struct(j).Phrase(end) = [];
                else
                    exclude = [exclude j];
                end
            elseif stim_type == 1
                if strcmp(Struct(j).Phrase(end),'1') && ismember(Struct(j).Phrase(1:end-1),wlist)
                    include = [include j];
                    
                    %Remove Tailing number from label if repetition
                    Struct(j).Phrase(end) = [];
                else
                    exclude = [exclude j];
                end
            end
        else
            if ismember(Struct(j).Phrase,wlist)
                include = [include j];
            else
                exclude = [exclude j];
            end
            
        end
    end
    
    disp(['Excluding ' num2str(length(exclude)) ' events, ' num2str(length(include)) ' remaining']);
    
    %Find labels
    notesi{i} = {Struct(include).Phrase};
    
    %Find sample dimensions and phoneme transitions
    phoneTrans{i} = cell(length(include),length(Struct(include(1)).Words));
    for j = 1:length(include)
        t = include(j);
        
        
        
        %Determine length of trial
        for ii = 1:length(suffix)
            EvntSampLength(j,ii) = eval(sprintf('size(Struct(k).%s,2);',suffix{ii}));
        end
        
        % populate list of phoneme transitions
        for k = 1:length(Struct(t).Words)
        %          disp(k);
            tmp = [Struct(t).Words(k).Phones]; stupidThing = [tmp.StopTime];
            phoneTrans{i}{j,k} = [tmp.StartTime stupidThing(end)];
        end
    end
    
    % data range to select
%     disp([num2str(size(phoneTrans{i},2)) ' phoneme transitions detected. Aligning to word number ' num2str(t_zero)])
    selWord = phoneTrans{i}(:,t_zero{1});
    for j = 1:length(selWord)
        thisWord = selWord{j};
        if t_zero{2} < 0
            thisWord = fliplr(thisWord);
            trans = abs(t_zero{2});
        else
            trans = t_zero{2};
        end
        Start(j,1) = thisWord(trans) - n_bef;
        if length(phoneTrans) == 1
            Stop(j,1) = thisWord(end) + n_aft;
        else
            Stop(j,1) = thisWord(trans) + n_aft;
        end
    end
    
    StartSamp = floor(repmat(Start,1,length(fs)).*repmat(fs,length(Start),1)) + 1;  %repmats are to handle multiple fs
    StopSamp = floor(repmat(Stop,1,length(fs)).*repmat(fs,length(Stop),1)) + 1;
    SampLength = StopSamp - StartSamp;
    maxSamp(i,:) = max(StopSamp - StartSamp);
    
    if any(any(StartSamp < 1)) || any(any((EvntSampLength - StopSamp)<0))
        error(['Data range in evnt structure not long enough to accomodate current timing parameters.'...
            'Choose smaller timing window, or recreate evnt structure with larger bef and aft.'])
    end
    
    %Calculate new Onset/Offset Times, absStarts
    OnsetTimesi{i} = bef - Start;
    OffsetTimesi{i} = phoneTrans{i}(:,end);
    AlignTimes(i,:) = n_bef;
    AbsStarti{i}= [Struct(include).AbsStart];
    AbsStopi{i} = [Struct(include).AbsStop];
    evntTriali{i} = include';
    blocki{i} = repmat(cellstr([subject '_B' num2str(blocks(i))]),size(include'));
    spatialID{i} = Struct(include).spatialID;
    
    %Prealocate Y Matrix
    for j = 1:length(fs)
        Y{i,j} = nan(numFeatures(j),maxSamp(i,j),length(include));
    end
    
    
    %Construct data matrix
    for j = 1:length(include)
        k = include(j);
        labsi{i}(j) = find(strcmp(notesi{i}{j}, wlist));
        %         range{i}(j) = round((StartTimesi{i}(j)-bef)*fs1):round((StartTimesi{i}(j)+aft)*fs1);
        for ii = 1:length(fs)
            SampRange{ii} = StartSamp(j,ii):StopSamp(j,ii);
        end
        if datatype == 3;
            if FT_only == 1
                Y{i}(:,1:SampLength(j)+1,j) = [Struct(k).Kin_data.lips.x(:,SampRange{ii}); Struct(k).Kin_data.lips.y(:,SampRange{ii});...
                    Struct(k).Kin_data.jaw.xy(:,SampRange{ii}); Struct(k).Kin_data.nose.xy(:,SampRange{ii})];
            elseif US_only == 1
                Y{i}(:,1:SampLength(j)+1,j) = [Struct(k).Kin_data.tongue.x(:,SampRange{ii}); Struct(k).Kin_data.tongue.y(:,SampRange{ii})];
            elseif FT_only == 0
                Y{i}(:,1:SampLength(j)+1,j) = [Struct(k).Kin_data.lips.x(:,SampRange{ii}); Struct(k).Kin_data.lips.y(:,SampRange{ii});...
                    Struct(k).Kin_data.jaw.xy(:,SampRange{ii}); Struct(k).Kin_data.nose.xy(:,SampRange{ii});...
                    Struct(k).Kin_data.tongue.x(:,SampRange{ii}); Struct(k).Kin_data.tongue.y(:,SampRange{ii})];
            end
%         elseif datatype == 9
%             for ii = 1: length(suffix)
%                 Y{i,ii}(:,:,1:SampLength(j,ii)+1,j) = Struct(k).AA_data.elctrd(:,:,SampRange{ii});
%             end
        else
            for ii = 1: length(suffix)
                eval(sprintf('Y{i,ii}(:,1:SampLength(j,ii)+1,j) = Struct(k).%s(:,SampRange{ii});', suffix{ii}));
            end
            
        end
        
    end
end

%Combine blocks into single matrix

for ii = 1:length(prefix)
    eval(sprintf('%sY = [];',lower(prefix{ii})));
end
meta.labs = [];  
meta.OnsetTimes = [];
meta.OffsetTimes = [];
meta.AbsStart = [];
meta.AbsStop = [];
meta.notes = [];
meta.block = [];
meta.evntTrial = [];

if datatype == 1 || datatype == 7 || datatype == 9
    meta.baselineMethod = [];
    meta.baselineTime = [];
    meta.baselineBlock = [];
    meta.baselineMu = [];
    meta.baselineSTD = [];
end

for i = 1:length(blocks)
    for ii = 1:length(prefix)
        tmp = cat(2,Y{i,ii}, nan(size(Y{i,ii},1),max(maxSamp(:,ii))+1 - size(Y{i,ii},2) ,size(Y{i,ii},3)));
        eval(sprintf('%sY = cat(3,%sY,tmp);',lower(prefix{ii}),lower(prefix{ii})));
    end
    meta.labs = [meta.labs; labsi{i}'];
    meta.OnsetTimes = [meta.OnsetTimes; OnsetTimesi{i}];
    meta.OffsetTimes = [meta.OffsetTimes; OffsetTimesi{i}];
    meta.AbsStart = [meta.AbsStart; AbsStarti{i}'];
    meta.AbsStop = [meta.AbsStop; AbsStopi{i}'];
    meta.notes = [meta.notes; notesi{i}'];
    meta.block = [meta.block; blocki{i}];
    meta.evntTrial = [meta.evntTrial; evntTriali{i}];
    if datatype == 1 || datatype == 7 || datatype == 9
        meta.baselineMethod = [meta.baselineMethod; repmat(cellstr(baselineMethod{i}),size(blocki{i}))];
        meta.baselineTime = [meta.baselineTime; repmat(baselineTime{i},size(blocki{i}))];
        meta.baselineBlock = [meta.baselineBlock; repmat(cellstr(baselineBlock{i}),size(blocki{i}))];
        meta.baselineMu = [meta.baselineMu; repmat(baselineMu{i},size(blocki{i}))];
        meta.baselineSTD = [meta.baselineSTD; repmat(baselineSTD{i},size(blocki{i}))];
%     if datatype == 9
%         meta.baselineMethod = [meta.baselineMethod; repmat(cellstr(baselineMethod{i}),size(blocki{i}))];
%         meta.baselineTime = [meta.baselineTime; repmat(baselineTime{i},size(blocki{i}))];
%         meta.baselineBlock = [meta.baselineBlock; repmat(cellstr(baselineBlock{i}),size(blocki{i}))];
%         meta.baselineMu = [meta.baselineMu; repmat(baselineMu{i},size(blocki{i}))];
%         meta.baselineSTD = [meta.baselineSTD; repmat(baselineSTD{i},size(blocki{i}))];
    end
end
meta.spatialID = Struct(1).spatialID;
meta.wlist = wlist;
meta.fs = fs;
%Save
bname = sprintf('%g_',blocks);
eval(sprintf('save ([subject ''.'' stim_type_id{stim_type} ''.%s.B'' bname(1:end-1) ''.align'' num2str(t_zero{1}) ''_'' num2str(t_zero{2}) ''.%s.mat''],''%sY'',''meta'',''-v7.3'');',tname,lower(prefix{:}),lower(prefix{:})));
end

