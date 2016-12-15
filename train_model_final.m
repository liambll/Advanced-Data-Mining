%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: llbui
% date: 27 Oct for release this code

clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data,wordMap]=read_data;
wordMap('<PAD>') = length(wordMap) + 1;
wordMap('<UNK>') = length(wordMap) + 1;

% section 1.2 seperate dataset to training and validation sets
min_length = 5;
train_length = length(data) * 0.8;
validation_length = length(data) - train_length;

data_indice = cell(length(data),2);
for i=1:length(data)
    sentence = data{i,2};
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
    label = data{i,3};
    
    indices = zeros(size(sentence));
    for j=1:length(indices)
        indices(j) = wordMap(sentence{j});
    end
    data_indice{i,1} = indices;
    
    if label == 1
        label = 1;
    else
        label = 2;
    end
    data_indice{i,2} = label;
end

data_indice = data_indice(randperm(length(data_indice)),:);
data_train = data_indice(1:train_length,:);
data_validation = data_indice(train_length+1:end,:);

% init word embeding
d = 100;
use_glove = 1;   

if use_glove == 1
    if (d ~= 50) && (d ~= 100) && (d ~= 200) && (d ~= 300)
        fprintf('GloVe can only be used with word dimension 50, 100, 200, 300. Proceeding without Glove...\n');
        use_glove = 0;
    else
        fprintf('Loading Glove...\n');
        glove = vector_representation(d);
    end
end
if use_glove == 1
    % read word vector from GloVe word embedding
    wordMap_key = wordMap.keys();
    T = zeros(length(wordMap),d);
    for i = 1:length(wordMap_key)
        if isKey(glove,wordMap_key{i})
            T(wordMap(wordMap_key{i}),:) = str2double(glove(wordMap_key{i}));
        else
            T(wordMap(wordMap_key{i}),:) = normrnd(0,0.1,[1,d]);
        end
    end
    clear glove;
else   
    % initialize word vector
    T = normrnd(0,0.1,[length(wordMap),d]);
end

% store best model
best_accuracy = 0;
best_w_out = 0;
best_b_out = 0;
best_w_conv = 0;
best_b_conv = 0;
best_T = 0;

retrain_T = 10;
for retrain = 1:retrain_T
    % init filter
    filter_size = [2,3,4,5];
    n_filter = 6;

    w_conv = cell(length(filter_size), 1);
    b_conv = cell(length(filter_size), 1);

    for i=1:length(filter_size)
        w_conv{i} = normrnd(0, 0.1, [filter_size(i), d, 1, n_filter]);
        b_conv{i} = zeros(n_filter, 1);
    end

    % init output layer
    total_filters = length(filter_size) * n_filter;
    n_class = 2;
    w_out = normrnd(0, 0.1, [total_filters, n_class]);
    b_out = zeros(n_class, 1);

    % learning parameters
    step_size = 1e-2; % learning rate
    reg = 1e-1; % regularization
    max_iter = 20;
    threshold = 0.0001; % early stop threshold
    prev_loss = 0;

    for iter=1:max_iter
        preds = zeros(train_length,1);
        actuals = zeros(train_length,1); 

    %% Section 2: training
    % Note: 
    % you may need the resouces [2-4] in the project description.
    % you may need the follow MatConvNet functions: 
    %       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()

        % for each example in train.txt do  
        for ind = 1:train_length

            % section 2.1 forward propagation and compute the loss
            %get sentence matrix
            word_indices = data_train{ind,1};
            y = data_train{ind, 2};
            X = T(word_indices,:);

            pool_res = cell(1, length(filter_size));
            cache = cell(2, length(filter_size));
            for i = 1: length(filter_size)
                %convolution operation
                conv = vl_nnconv(X, w_conv{i}, b_conv{i});

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


            % compute output layer
            o = vl_nnconv(reshape(z, [total_filters,1]), reshape(w_out,[total_filters,1,1,2]), b_out);
            actuals(i) = y;
            [~, preds(i)] = max(o);

            % compute loss
            loss = vl_nnloss(o, y);

            % section 2.2 backward propagation and compute the derivatives
            do = vl_nnloss(o, y, 1);
            [dz,dw_out, db_out] = vl_nnconv(reshape(z, [total_filters,1]), reshape(w_out,[total_filters,1,1,2]), b_out,do);       
            dpool_res = vl_nnconcat(pool_res, 3, reshape(dz, [1,1,total_filters]));
            cache_conv = cell(3, length(filter_size));
            for i = 1: length(filter_size)
                dpool = dpool_res{i};
                sizes = size(cache{1,i});
                drelu = vl_nnpool(cache{2,i}, [sizes(1), 1], dpool);
                dconv = vl_nnrelu(cache{1,i}, drelu);
                [dx, dw_conv, db_conv] = vl_nnconv(X, w_conv{i}, b_conv{i}, dconv);

                % keep value for  parameters update
                cache_conv{1,i} = dx;
                cache_conv{2,i} = dw_conv;
                cache_conv{3,i} = db_conv;
            end

            % section 2.3 update the parameters
            % update filter and word vector
            for i=1:length(filter_size)
                X = X - step_size*cache_conv{1,i};
                w_conv{i} = w_conv{i} - step_size*cache_conv{2,i};
                b_conv{i} = b_conv{i} - step_size*cache_conv{3,i};
            end
            for i=1:length(word_indices)
                T(word_indices(i),:) = X(i,:);
            end   

            % update output layer
            w_out = w_out - step_size*reshape(dw_out,[total_filters, n_class]);
            b_out = b_out - step_size*db_out;
        end
    %% Section 3: evaluate prediction
        % approximate train accuracy. This does not rerun for training set
        accuracy = length(find(preds==actuals))/train_length;
        fprintf('iter: %d - train accuracy: %f, ', iter, accuracy);

        % validation accuracy
        total_loss = 0;
        correct = 0;
        for ind = 1:validation_length
            %get sentence matrix
            word_indices = data_validation{ind,1};
            y = data_validation{ind, 2};
            X =T(word_indices,:);

            pool_res = cell(1, length(filter_size));
            cache = cell(2, length(filter_size));
            for i = 1: length(filter_size)
                %convolution operation
                conv = vl_nnconv(X, w_conv{i}, b_conv{i});

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
            o = vl_nnconv(reshape(z, [total_filters,1]), reshape(w_out,[total_filters,1,1,2]), b_out);
            [~,pred] = max(o);
            if pred == y
                correct = correct + 1;
            end
            
            % loss
            loss = vl_nnloss(o, y);
            total_loss = total_loss + abs(loss);
            
        end

        accuracy = correct/validation_length;
        total_loss = total_loss/validation_length;
        fprintf('validation accuracy: %f , validation loss: %f\n', accuracy, total_loss);
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_w_out = w_out;
            best_b_out = b_out;
            best_w_conv = w_conv;
            best_b_conv = b_conv;
            best_T = T;  
        end
        
        % stop if loss difference is less than threshold
        if abs(total_loss - prev_loss) < threshold
            fprintf('Early stop... \n');
            prev_loss = 0;
            break;
        end
        prev_loss = total_loss;
    end
    fprintf('Retrain %d: best validation accuracy: %f \n', retrain, best_accuracy);
end