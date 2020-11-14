clc;
clear;
%rand('state',2);

%% pos
data = csvread('./301-untar-pos.csv');
label = data(1,1:end);
label = label';
data = data(2:end,1:end);
data = data';
idx=1:100;
val_data = data(idx,:);
val_label = label(idx,:);


data1 = csvread('./beida-untar-pos.csv');
label1 = data1(1,1:end);
label1 = label1';
data1 = data1(2:end,1:end);
data1 = data1';

data2 = csvread('./3yuan-untar-pos.csv');
label2 = data2(1,1:end);
label2 = label2';
data2 = data2(2:end,1:end);
data2 = data2';

tmp_label = [label;label1;label2];
tmp_data = [data;data1;data2];
tmp_data = sqrt(tmp_data);

val_label = [val_label];
val_data = [val_data];
val_data = sqrt(val_data);

addpath('./liblinear-2.20/matlab/');
fprintf('train process ... ... \n');
option=['-c 5'];
pos_data = tmp_data;
pos_label = tmp_label;
pos_val_data = val_data;
pos_val_label = val_label;

if 1
    weights=[];
    all_acc=[];
    all_score1=[];
    for r=1:5000
        r
        train_data=[];
        train_label=[];
        test_data=[];
        test_label=[];

        neg_idx = find(pos_label==0);
        len = length(neg_idx);
        rand_idx = randperm(len);
        neg_idx = neg_idx(rand_idx);
        for i=1:length(neg_idx)
            if mod(i,4)==0
                test_data=[test_data;pos_data(neg_idx(i),:)];
                test_label =[test_label;pos_label(neg_idx(i))];
            else
                 train_data=[train_data;pos_data(neg_idx(i),:)];
                 train_label =[train_label;pos_label(neg_idx(i))];        
             end
        end

        pos_idx = find(pos_label==1);
        len=length(pos_idx);
        rand_idx = randperm(len);
        pos_idx = pos_idx(rand_idx);
        for i=1:length(pos_idx)
            if mod(i,4)==0
                test_data=[test_data;pos_data(pos_idx(i),:)];
                test_label =[test_label;pos_label(pos_idx(i))];
            else
                train_data=[train_data;pos_data(pos_idx(i),:)];
                train_label =[train_label;pos_label(pos_idx(i))];        
             end
        end  
        model = train(train_label,sparse(normr(train_data(:,:))),option);
        [p1,c1,d1] = predict(train_label,sparse(normr(train_data(:,:))),model);
        stats1=confusionmatStats(train_label,p1);
        [p2,c2,d2] = predict(test_label,sparse(normr(test_data(:,:))),model);
        stats2=confusionmatStats(test_label,p2);
        [p3,c3,d3] = predict(pos_val_label,sparse(normr(pos_val_data)),model);
        stats3=confusionmatStats(pos_val_label,p3);
        all_score1=[all_score1;c1(1),c2(1),c3(1),stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats2.specificity(2),stats2.sensitivity(2),c2(1)/100,stats3.specificity(2),stats3.sensitivity(2),c3(1)/100];
        svm_weight = model.w;
        svm_weight = svm_weight.^2;
        all_acc=[all_acc;c1(1),c2(1),c3(1)];
        weights=[weights;svm_weight];
    end
    [muhat,sigmahat,muci,sigmaci]=normfit(all_score1);
    xlswrite('./data/all_feature_pos_weight.xlsx',weights);
    tmp = [muhat,muci(1,:),muci(2,:)];
    xlswrite('./data/all_feature_pos_data_analysis.xlsx',tmp);
    xlswrite('./data/all_feature_pos_all_score.xlsx',all_score1);
end


%% neg
data = csvread('./301-untar-neg.csv');
label = data(1,1:end);
label = label';
data = data(2:end,1:end);
data = data';
idx=1:100;
val_data = data(idx,:);
val_label = label(idx,:);


data1 = csvread('./beida-untar-neg.csv');
label1 = data1(1,1:end);
label1 = label1';
data1 = data1(2:end,1:end);
data1 = data1';

data2 = csvread('./3yuan-untar-neg.csv');
label2 = data2(1,1:end);
label2 = label2';
data2 = data2(2:end,1:end);
data2 = data2';

tmp_label = [label;label1;label2];
tmp_data = [data;data1;data2];
tmp_data = sqrt(tmp_data);

val_label = [val_label];
val_data = [val_data];
val_data = sqrt(val_data);

neg_data=tmp_data;
neg_label = tmp_label;
neg_val_data = val_data;
neg_val_label = val_label;

%%
if 1
    weights=[];
    all_acc=[];
    all_score2=[];
    for r=1:5000
        r
        train_data=[];
        train_label=[];
        test_data=[];
        test_label=[];

        neg_idx = find(neg_label==0);
        len = length(neg_idx);
        rand_idx = randperm(len);
        neg_idx = neg_idx(rand_idx);
        for i=1:length(neg_idx)
            if mod(i,4)==0
                test_data=[test_data;neg_data(neg_idx(i),:)];
                test_label =[test_label;neg_label(neg_idx(i))];
            else
                 train_data=[train_data;neg_data(neg_idx(i),:)];
                 train_label =[train_label;neg_label(neg_idx(i))];        
             end
        end

        pos_idx = find(neg_label==1);
        len=length(pos_idx);
        rand_idx = randperm(len);
        pos_idx = pos_idx(rand_idx);
        for i=1:length(pos_idx)
            if mod(i,4)==0
                test_data=[test_data;neg_data(pos_idx(i),:)];
                test_label =[test_label;neg_label(pos_idx(i))];
            else
                train_data=[train_data;neg_data(pos_idx(i),:)];
                train_label =[train_label;neg_label(pos_idx(i))];        
             end
        end  
        model = train(train_label,sparse(normr(train_data(:,:))),option);
        [p1,c1,d1] = predict(train_label,sparse(normr(train_data(:,:))),model);
        stats1=confusionmatStats(train_label,p1);
        [p2,c2,d2] = predict(test_label,sparse(normr(test_data(:,:))),model);
        stats2=confusionmatStats(test_label,p2);
        [p3,c3,d3] = predict(neg_val_label,sparse(normr(neg_val_data)),model);
        stats3=confusionmatStats(neg_val_label,p3);
        all_score2=[all_score2;c1(1),c2(1),c3(1),stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats2.specificity(2),stats2.sensitivity(2),c2(1)/100,stats3.specificity(2),stats3.sensitivity(2),c3(1)/100];
        svm_weight = model.w;
        svm_weight = svm_weight.^2;
        all_acc=[all_acc;c1(1),c2(1),c3(1)];
        weights=[weights;svm_weight];
    end
    [muhat,sigmahat,muci,sigmaci]=normfit(all_score2);
    xlswrite('./data/all_feature_neg_weight.xlsx',weights);
    tmp = [muhat,muci(1,:),muci(2,:)];
    xlswrite('./data/all_feature_neg_data_analysis.xlsx',tmp);
    xlswrite('./data/all_feature_neg_all_score.xlsx',all_score2);
   
end

