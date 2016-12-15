%% Predict sentiments
% author: llbui
% date: 27 Oct for release this code

%% Section 1: read test data
% section 1.1 read file 'sample_test.txt' and load data
fileName = 'sample_test.txt';
headLine = true;
separater = '::';

data_test = cell(1000, 2);

fid = fopen(fileName, 'r');
line = fgets(fid);

ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);
    sid = str2double(attrs{1});
    
    s = attrs{2};
    w = strsplit(s);
    
    % save data
    data_test{ind, 1} = sid;
    data_test{ind, 2} = w;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end

% section 1.2 handle data
min_length = 5;
data_test_indices = cell(length(data_test));

for i=1:length(data_test)
    sentence = data_test{i,2};
    sentence_length = length(sentence);
    % pad sentence shorter than filter length
    if sentence_length < min_length
        sentence2 = cell(1, min_length);
        for j=1:sentence_length
            sentence2{j} = sentence{j};
        end
        for j=(sentence_length+1):min_length
            sentence2{j} = '<PAD>';
        end
        sentence = sentence2;
    end
    
    indices = zeros(size(sentence));
    for j=1:length(indices)
        if isKey(wordMap,sentence{j})
            indices(j) = wordMap(sentence{j});
        else
            indices(j) = wordMap('<UNK>');
        end
    end
    data_test_indices{i} = indices;
end

%% Section 2: apply model 
output = cell(length(data_test),2);
for ind = 1:length(data_test) 
    %get sentence matrix
    word_indices = data_test_indices{ind,1};
    X =best_T(word_indices,:);

    pool_res = cell(1, length(filter_size));
    cache = cell(2, length(filter_size));
    for i = 1: length(filter_size)
        %convolution operation
        conv = vl_nnconv(X, best_w_conv{i}, best_b_conv{i});

        %activation reLu
        relu = vl_nnrelu(conv);

        % 1-max pooling
        sizes = size(conv);
        pool = vl_nnpool(relu, [sizes(1), 1]);

        % keep values for back-propagate
        cache{2,i} = relu;
        cache{1,i} = conv;
        pool_res{i} = pool;
    end

    % concatenate
    z = vl_nnconcat(pool_res, 3);

    % ouput layer
    o = vl_nnconv(reshape(z, [total_filters,1]), reshape(best_w_out,[total_filters,1,1,2]), best_b_out);
    [~,pred] = max(o);

    % output
    output{ind,1} = data_test{ind,1};
    if pred == 1
        output{ind,2} = 1;
    else
        output{ind,2} = 0;
    end
end

%% Section 3: write output to file
textHeader = 'id::label'; % header
fid = fopen(strcat('submission_',fileName),'w'); 
fprintf(fid,'%s\n',textHeader);
for ind = 1:length(output)
    fprintf(fid,'%i::%i\n',output{ind,1},output{ind,2});
end
fclose(fid);
