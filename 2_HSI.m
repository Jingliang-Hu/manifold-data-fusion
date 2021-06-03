%% read enmap
% LCLU_Berlin data directory path:
dPath = './data/LCLU_Berlin';
sysPath = '.';
addpath(genpath(sysPath));
% read training and testing label
load([dPath,'/trainAndTest.mat'])
% read LS8 features
load([dPath,'/enmap_berlin.mat'], 'hsi_up_mask')
load([dPath,'/enmap_berlin.mat'], 'mask')

EnMapFeat = hsi_up_mask; clear hsi_up_mask
testMap = double(testMap);


%% Feature extraction (PCA MP & zero-one normalization)
temp = reshape(EnMapFeat,size(trainMap,1)*size(trainMap,2),size(EnMapFeat,3));
normData = temp(mask(:)==1,:);
for i = 1:size(normData,2)
    normData(:,i) = mat2gray(normData(:,i));
end
normData = pca_dr(normData);
temp = zeros(size(temp,1),size(normData,2));
temp(mask(:)==1,:) = normData;

PolFeat = reshape(temp,size(trainMap,1),size(trainMap,2),size(temp,2));

r1 = 1:3;
r2 = r1;
options = 'MPr';
temp=[];
for i=1:size(PolFeat,3)
   MPNeachpca = Make_Morphology_profile(PolFeat(:,:,i),r1,r2,options);
   temp = cat(3,temp,MPNeachpca);        
end

temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3));

trainFeat = temp(trainMap(:)>0,:);
trainLab = trainMap(trainMap(:)>0);

testFeat = temp(testMap(:)>0,:);
testLab = testMap(testMap(:)>0);

%% Classification

% knn
rng(1)
mdl =ClassificationKNN.fit(trainFeat,trainLab,'NumNeighbors',1,'distance','euclidean'); 
predLab= predict(mdl,testFeat); 
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI KNN: ', num2str(round(oa*1e4)/1e2),'%'])


% linear svm 
rng(1)
t = templateSVM('Standardize',0,'Solver','SMO','KernelFunction','linear','KernelScale','auto');
L_SVM1 = fitcecoc(trainFeat,trainLab,'Learners',t);
predLab = predict(L_SVM1,testFeat);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI Linear SVM: ', num2str(round(oa*1e4)/1e2),'%'])


% rbf - svm
rng(1)
folds = 5;
[ cv_C, cv_G, ~ ] = kenelPar( trainLab,trainFeat,folds );% optimal parameters searching
K_SVM1 = svmtrain(trainLab,trainFeat,sprintf('-c %f -g %f', cv_C, cv_G));% SVM training
[predLab] = svmpredict(testLab, testFeat, K_SVM1);% predict using trained model
[ M,oa,pa,ua,kappa ] = confusionMatrix( testLab, predLab );
display(['Overall accuracy of HSI Gaussian kernel SVM: ', num2str(round(oa*1e4)/1e2),'%'])



% rf
rng(1); % For reproducibility
NumTrees = 40;
Mdl_rf = TreeBagger(NumTrees,trainFeat,trainLab);
predLab = predict(Mdl_rf,testFeat);
predLab = cellfun(@str2double,predLab);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI random forest: ', num2str(round(oa*1e4)/1e2),'%'])



% ccf
rng(1); % For reproducibility
nb_trees = 40;
[ccfs] = genCCF(nb_trees,trainFeat,trainLab);
[predLab, ~, ~] = predictFromCCF(ccfs,testFeat);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI canonical correlation forest: ', num2str(round(oa*1e4)/1e2),'%'])
