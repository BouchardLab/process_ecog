function baselineD = CalcBaseline(subject,block,baselineTime,elects,frqs,method)

bdir = [subject '_B' num2str(block)];
ddir = [bdir '/HilbAA_70to150_8band'];

if exist([bdir '/' subject '_B' num2str(block) '_transcription_final.TextGrid'])
    transcription = [subject '_B' num2str(block) '_transcription_final.TextGrid'];
    [evnt,badsegs] = ParseTextGrid([bdir '/' transcription]); %extract times of events and remove bad times
    
elseif exist([bdir '/' subject '_B' num2str(block) '_transcription_final.lab'])
    transcription = [subject '_B' num2str(block) '_transcription_final.lab'];
    [evnt,badsegs] = ParseLab([bdir '/' transcription]); %extract times of events and remove bad times
else
    error('Transcript File Not Found')
end

if exist([bdir '/' 'baselineTime.mat'])
    load([bdir '/' 'baselineTime.mat'])
end

badChannels = [];
badChannels = load([bdir '/Artifacts/badChannels.txt']);


for cnt1 = 1:length(elects)
    temp = num2str(mod(elects(cnt1),64)); %recordings are organized into 4x64 chn blocks
    if strcmpi(temp,'0'), temp = '64';end
    fname = ['Wav' num2str(ceil(elects(cnt1)/64))  temp '.htk'];
    [tempdata,fs] = readhtk([ddir '/' fname]);
    
    tol = .5; %%% +/- around bad segments
    goodinds = 1:length(tempdata);
    if ~isempty(badsegs)        
        for cnt2 = size(badsegs,1):-1:1
            stop = badsegs(cnt2,2)+tol;
            start = badsegs(cnt2,1)-tol;
            if start < 0
                start = 1/fs;
            elseif stop > length(tempdata)
                stop = length(tempdata)/400;
            end
            goodinds(1+floor(start*fs):ceil(stop*fs)) = NaN;
        end
        %goodinds = [goodinds (1+floor(start*fs)):length(tempdata)];
    end
    tmp1 = ceil(baselineTime(1)*fs);
    goodinds = goodinds(goodinds>tmp1);
    tmp1 = ceil(baselineTime(2)*fs);
    goodinds(goodinds>tmp1) = [];
    
    
    baselineD(1).mu_std(cnt1,1) = mean(mean(tempdata(frqs,goodinds)));
    baselineD(1).mu_std(cnt1,2) = std(mean(tempdata(frqs,goodinds)));
end
baselineBlock = bdir;
save([bdir '/baseline.HG.mat'],'baselineD','baselineTime','method','baselineBlock')


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
