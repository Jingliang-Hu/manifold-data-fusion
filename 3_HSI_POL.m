restoredefaultpath
addpath(genpath('.'));

%% read training and testing label
load('.\data\LCLU_Berlin\trainAndTest.mat')
testMap = double(testMap);
%% read sentinel 1 data and feature extraction
load('.\data\LCLU_Berlin\sentinel1_berlin_C.mat', 'C')
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'mask')
polsar = C;clear C

coh = sqrt(polsar(:,:,3).^2+polsar(:,:,4).^2)./sqrt(polsar(:,:,1).*polsar(:,:,2));
ratio = (polsar(:,:,1)./polsar(:,:,2));
PolFeat = cat(3,polsar(:,:,1:2),ratio,coh);

% MP
r1 = 1:3;
r2 = r1;
options = 'MPr';
POLMP=[];
for i=1:size(PolFeat,3)
   MPNeachpca = Make_Morphology_profile(PolFeat(:,:,i),r1,r2,options);
   POLMP = cat(3,POLMP,MPNeachpca);        
end

% statistics
[ featMean ] = localStat( PolFeat,5,'mean' );
[ featStd ]  = localStat( PolFeat,5,'std' );
POLMP = cat(3,POLMP,featMean,featStd);   
PolFeat = reshape(POLMP,size(trainMap,1)*size(trainMap,2),size(POLMP,3));




%% read EnMAP features and feature extraction
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'hsi_up_mask')
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'mask')
HSIFeat = hsi_up_mask; clear hsi_up_mask

temp = reshape(HSIFeat,size(trainMap,1)*size(trainMap,2),size(HSIFeat,3));
normData = temp(mask(:)==1,:);
for i = 1:size(normData,2)
    normData(:,i) = mat2gray(normData(:,i));
end
normData = pca_dr(normData);
HSIFeat = zeros(size(temp,1),size(normData,2));
HSIFeat(mask(:)==1,:) = normData;


HSIFeat = reshape(HSIFeat,size(trainMap,1),size(trainMap,2),size(HSIFeat,2));

r1 = 1:3;
r2 = r1;
options = 'MPr';
temp=[];
for i=1:size(HSIFeat,3)
   MPNeachpca = Make_Morphology_profile(HSIFeat(:,:,i),r1,r2,options);
   temp = cat(3,temp,MPNeachpca);        
end

HSIFeat = reshape(temp,size(temp,1)*size(temp,2),size(temp,3));

feat = cat(2,HSIFeat,PolFeat);

%% classification (zero-one normalization)
temp = feat;
normData = temp(mask(:)==1,:);
for i = 1:size(normData,2)
    normData(:,i) = mat2gray(normData(:,i));
end
temp(mask(:)==1,:) = normData;

trainFeat = temp(trainMap(:)>0,:);
trainLab = trainMap(trainMap(:)>0);

testFeat = temp(testMap(:)>0,:);
testLab = testMap(testMap(:)>0);

% knn
rng(1)
mdl =ClassificationKNN.fit(trainFeat,trainLab,'NumNeighbors',1,'distance','euclidean'); 
predLab= predict(mdl,testFeat); 
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI+POL KNN: ', num2str(round(oa*1e4)/1e2),'%'])


% linear svm 
rng(1)
t = templateSVM('Standardize',0,'Solver','SMO','KernelFunction','linear','KernelScale','auto');
L_SVM1 = fitcecoc(trainFeat,trainLab,'Learners',t);
predLab = predict(L_SVM1,testFeat);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI+POL Linear SVM: ', num2str(round(oa*1e4)/1e2),'%'])


% rbf - svm
rng(1)
folds = 5;
[ cv_C, cv_G, ~ ] = kenelPar( trainLab,trainFeat,folds );% optimal parameters searching
K_SVM1 = svmtrain(trainLab,trainFeat,sprintf('-c %f -g %f', cv_C, cv_G));% SVM training
[predLab] = svmpredict(testLab, testFeat, K_SVM1);% predict using trained model
[ M,oa,pa,ua,kappa ] = confusionMatrix( testLab, predLab );
display(['Overall accuracy of HSI+POL Gaussian kernel SVM: ', num2str(round(oa*1e4)/1e2),'%'])


% rf
rng(1); % For reproducibility
NumTrees = 40;
Mdl_rf = TreeBagger(NumTrees,trainFeat,trainLab,'OOBPredictorImportance','on');
figure,bar(Mdl_rf.OOBPermutedVarDeltaError)
predLab = predict(Mdl_rf,testFeat);
predLab = cellfun(@str2double,predLab);
[ M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI+POL random forest: ', num2str(round(oa*1e4)/1e2),'%'])

% ccf
rng(1); % For reproducibility
nb_trees = 40;
[ccfs] = genCCF(nb_trees,trainFeat,trainLab);
[predLab, ~, ~] = predictFromCCF(ccfs,testFeat);
[M,oa,pa,ua,kappa ] = confusionMatrix(testLab,predLab);
display(['Overall accuracy of HSI+POL canonical correlation forest: ', num2str(round(oa*1e4)/1e2),'%'])
