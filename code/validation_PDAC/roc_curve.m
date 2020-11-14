function auc = roc_curve(deci,label_y,train)
    label_y=(label_y-0.5)*2;
    %%
    val0=0.0;
    cutoff = (val0-min(deci))/(max(deci)-min(deci));
    %%
    deci = (deci-min(deci))/(max(deci)-min(deci));
    [val,ind]=sort(deci,'descend');
    %%
    cutoff_idx = max(find(val>cutoff));
    %%
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y==-1)/sum(roc_y==-1);
    stack_y = cumsum(roc_y==1)/sum(roc_y==1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
    hold on
    plot(stack_x,stack_y);
    hold off
    hold on
    plot(stack_x(cutoff_idx),stack_y(cutoff_idx),'r*')
    hold off

    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    if train==1
        title(['Training ROC curve of (AUC=' num2str(auc) ')']);
        sprintf('Training ROC Zero Value: (%f %f)',stack_x(cutoff_idx),stack_y(cutoff_idx))
    else
        title(['Testing ROC curve of (AUC=' num2str(auc) ')']);
        sprintf('Testing ROC Zero Value: (%f %f)',stack_x(cutoff_idx),stack_y(cutoff_idx))
    end
end