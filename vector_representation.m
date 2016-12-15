function glove = vector_representation(d)
% Read vector representation of word from pre-trained word vector GloVe
%
% return: 
%       word vector GloVe

headLine = true;
fid = fopen(strcat('glove.6B.',num2str(d),'d.txt'), 'r');
line = fgets(fid);
separater = ' ';
attrs = strsplit(line, separater);
word = attrs{1};
vector = attrs(2:length(attrs));

glove = containers.Map;
ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    
    attrs = strsplit(line, separater);
    word = attrs{1};
    vector = attrs(2:length(attrs));
    
    % save data
    glove(word) = vector;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
fprintf('finish loading word vector GloVe\n');