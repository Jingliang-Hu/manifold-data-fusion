restoredefaultpath
addpath(genpath('.'));

%% Load Sentinel-1 data
% read training and testing label
load('.\data\LCLU_Berlin\trainAndTest.mat')
% read sentinel 1 data
load('.\data\LCLU_Berlin\sentinel1_berlin_C.mat', 'C')
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'mask')
testMap = double(testMap);
polsar = C;clear C


%% Polarimetric feature: Polsar elements from covariance matrix
coh = sqrt(polsar(:,:,3).^2+polsar(:,:,4).^2)./sqrt(polsar(:,:,1).*polsar(:,:,2));
ratio = (polsar(:,:,1)./polsar(:,:,2));
PolFeat = cat(3,polsar(:,:,1:2),ratio,coh);

%% Spatial feature: morphological profile
r1 = 1:3;
r2 = r1;
options = 'MPr';
MP =[];
for i=1:size(PolFeat,3)
   MPNeachpca = Make_Morphology_profile(PolFeat(:,:,i),r1,r2,options);
   MP = cat(3,MP,MPNeachpca);        
end

%% Statistical feature: mean and std of 11 by 11 sliding window
[ featMean ] = localStat( PolFeat,5,'mean' );
[ featStd ]  = localStat( PolFeat,5,'std' );

%% Concatenate features
feat = cat(3,MP,featMean,featStd);   

%% CLASSIFICATION
feat = reshape(feat,size(trainMap,1)*size(trainMap,2),size(feat,3));
trainFeat = feat(trainMap(:)>0,:);
trainLab = trainMap(trainMap(:)>0);

testFeat = feat(testMap(:)>0,:);
testLab = testMap(testMap(:)>0);

% knn
rng(1)
mdl =ClassificationKNN.fit(trainFeat,trainLab,'NumNeighbors',1,'distance','euclidean'); 
predLab= predict(mdl,testFeat); 
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of POL KNN: ', num2str(round(oa*1e4)/1e2),'%'])


% linear svm 
rng(1)
t = templateSVM('Standardize',0,'Solver','SMO','KernelFunction','linear','KernelScale','auto');
L_SVM1 = fitcecoc(trainFeat,trainLab,'Learners',t);
predLab = predict(L_SVM1,testFeat);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of POL Linear SVM: ', num2str(round(oa*1e4)/1e2),'%'])


% rbf - svm
rng(1)
folds = 5;
[ cv_C, cv_G, ~ ] = kenelPar( trainLab,trainFeat,folds );% optimal parameters searching
K_SVM1 = svmtrain(trainLab,trainFeat,sprintf('-c %f -g %f', cv_C, cv_G));% SVM training
[predLab] = svmpredict(testLab, testFeat, K_SVM1);% predict using trained model
[ M,oa,pa,ua,kappa ] = confusionMatrix( testLab, predLab );
display(['Overall accuracy of POL Gaussian kernel SVM: ', num2str(round(oa*1e4)/1e2),'%'])



% rf
rng(1); % For reproducibility
NumTrees = 40;
Mdl_rf = TreeBagger(NumTrees,trainFeat,trainLab,'OOBPredictorImportance','on');
figure,bar(Mdl_rf.OOBPermutedVarDeltaError)
predLab = predict(Mdl_rf,testFeat);
predLab = cellfun(@str2double,predLab);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of POL random forest: ', num2str(round(oa*1e4)/1e2),'%'])


% ccf
rng(1); % For reproducibility
nb_trees = 40;
[ccfs] = genCCF(nb_trees,trainFeat,trainLab);
[predLab, ~, ~] = predictFromCCF(ccfs,testFeat);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of POL canonical correlation forest: ', num2str(round(oa*1e4)/1e2),'%'])
