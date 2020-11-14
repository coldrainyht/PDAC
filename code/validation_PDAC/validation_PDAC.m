clc;
clear;
rand('state',2);

train_data = csvread('./Train.csv');
train_label = train_data(1,1:end);
train_label = train_label';
train_data = train_data(2:end,1:end);
train_data = train_data';
train_data = sqrt(train_data);


independent_data = csvread('./Independent_validaiton.csv');
independent_label = independent_data(1,1:end);
independent_label = independent_label';
independent_data = independent_data(2:end,1:end);
independent_data = independent_data';
independent_data = sqrt(independent_data);


val_data_1 = csvread('./Prospective_clinical_1.csv');
val_label_1 = val_data_1(1,1:end);
val_label_1 = val_label_1';
val_data_1 = val_data_1(2:end,1:end);
val_data_1 = val_data_1';
val_data_1 = sqrt(val_data_1);


val_data_2 = csvread('./Prospective_clinical_2.csv');
val_label_2 = val_data_2(1,1:end);
val_label_2 = val_label_2';
val_data_2 = val_data_2(2:end,1:end);
val_data_2 = val_data_2';
val_data_2 = sqrt(val_data_2);


diff = mean(train_data(211:end,:))./mean(train_data(1:210,:));
train_data(1:210,:) = train_data(1:210,:).*diff;

diff = mean(train_data(211:end,:))./mean(val_data_2(1:86,:));
val_data_2(1:86,:) = val_data_2(1:86,:).*diff;

diff = mean(train_data(211:end,:))./mean(val_data_1);
val_data_1 = val_data_1.*diff;

diff = mean(train_data(211:end,:))./mean(val_data_2(87:end,:));
val_data_2(87:end,:) = val_data_2(87:end,:).*diff;

diff = mean(train_data(211:end,:))./mean(independent_data(:,:));
independent_data = independent_data(:,:).*diff;


idx=[9,10,14,15,16,18,22,23,24,25,29,31,43,45,46,47,53,54,63,64,65,67,68,69,75,76,79,80,82,83,84,88,90,91,93,99,100,104:114,117,118,120,122,123,126,132,136,137,138,139,142,143,144,146,147,149,155,156,158,159,161];
val_data_2 = val_data_2(idx,:);
val_label_2 = val_label_2(idx);


addpath('./liblinear-2.20/matlab/');
fprintf('train process ... ... \n');
option=['-c 10'];


all_acc=[];
model = train(train_label(101:end),sparse(normr(train_data(101:end,:))),option);
[p1,c1,d1] = predict(train_label(101:end),sparse(normr(train_data(101:end,:))),model);
[p2,c2,d2] = predict(train_label(1:100),sparse(normr(train_data(1:100,:))),model);
[p3,c3,d3] = predict(independent_label,sparse(normr(independent_data(:,:))),model);
[p4,c4,d4] = predict([val_label_1;val_label_2],[sparse(normr(val_data_1(:,:)));sparse(normr(val_data_2(:,:)))],model);
all_acc=[all_acc;c1(1),c2(1)];
all_acc_2=[all_acc;c1(1),c3(1)];
all_acc_3=[all_acc;c1(1),c4(1)];


stats1=confusionmatStats(train_label(101:end),p1);
stats2=confusionmatStats(train_label(1:100),p2);
stats3=confusionmatStats(independent_label,p3);
stats4=confusionmatStats([val_label_1;val_label_2],[p4]);

tmp_score1=[stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats2.specificity(2),stats2.sensitivity(2),c2(1)/100]
tmp_score2=[stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats3.specificity(2),stats3.sensitivity(2),c3(1)/100]
tmp_score3=[stats1.specificity(2),stats1.sensitivity(2),c1(1)/100,stats4.specificity(2),stats4.sensitivity(2),c4(1)/100]


figure;
auc1 = roc_curve(d1,train_label(101:end),1);

figure;
auc2 = roc_curve(d2,train_label(1:100),0);

figure;
auc4 = roc_curve(d3,independent_label,0); 


figure;
auc3 = roc_curve([d4],[val_label_1;val_label_2],0);



    train_data = csvread('./CA19_9.csv');
    label_y = train_data(:,2);
    deci = train_data(:,1);
    label_y=(label_y-0.5)*2;
    mark = (37-min(deci))/(max(deci)-min(deci));
    deci = (deci-min(deci))/(max(deci)-min(deci));
    [val,ind]=sort(deci,'descend');
    point_inds=max(find(val>mark));
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y==-1)/sum(roc_y==-1);
    stack_y = cumsum(roc_y==1)/sum(roc_y==1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
    hold on
    plot(stack_x,stack_y);
    hold off
    
    hold on
    plot(stack_x(point_inds),stack_y(point_inds),'r*')
    hold off
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC=' num2str(auc3) ')']);
    
    generate_label = zeros(size(label_y));
    idx = find(train_data>37);
    generate_label(idx)=1;

    
