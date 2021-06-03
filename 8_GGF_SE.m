restoredefaultpath
addpath(genpath('.'));

%% read training and testing label
load('.\data\LCLU_Berlin\trainAndTest.mat')
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'mask')
testMap = double(testMap);

%% read enmap features
load('.\data\LCLU_Berlin\enmap_berlin.mat', 'hsi_up_mask')
EnMapFeat = hsi_up_mask; clear hsi_up_mask

HSIFeat = reshape(EnMapFeat,size(trainMap,1)*size(trainMap,2),size(EnMapFeat,3));
hsiNormData = HSIFeat(mask(:)==1,:);
for i = 1:size(hsiNormData,2)
    hsiNormData(:,i) = mat2gray(hsiNormData(:,i));
end
hsiNormData = pca_dr(hsiNormData);
HSIFeat = zeros(size(HSIFeat,1),size(hsiNormData,2));
HSIFeat(mask(:)==1,:) = hsiNormData;


HSIFeat = reshape(HSIFeat,size(trainMap,1),size(trainMap,2),size(HSIFeat,2));

r1 = 1:3;
r2 = r1;
options = 'MPr';
temp=[];
for i=1:size(HSIFeat,3)
   MPNeachpca = Make_Morphology_profile(HSIFeat(:,:,i),r1,r2,options);
   temp = cat(3,temp,MPNeachpca);        
end

temp = reshape(temp,size(temp,1)*size(temp,2),size(temp,3));

HSINormData = temp(mask(:)==1,:);
for i = 1:size(HSINormData,2)
    HSINormData(:,i) = mat2gray(HSINormData(:,i));
end
HSIFeat = zeros(size(temp,1),size(HSINormData,2));
HSIFeat(mask(:)==1,:) = HSINormData;


%% read sentinel-1 feature
load('.\data\LCLU_Berlin\sentinel1_berlin_C.mat', 'C')

coh = sqrt(C(:,:,3).^2+C(:,:,4).^2)./sqrt(C(:,:,1).*C(:,:,2));
ratio = (C(:,:,1)./C(:,:,2));
PolFeat = cat(3,C(:,:,1:2),ratio,coh);

% statistics
[ featMean ] = localStat( PolFeat,5,'mean' );
[ featStd ]  = localStat( PolFeat,5,'std' );


r1 = 1:3;
r2 = r1;
options = 'MPr';
temp=[];
for i=1:size(PolFeat,3)
   MPNeachpca = Make_Morphology_profile(PolFeat(:,:,i),r1,r2,options);
   temp = cat(3,temp,MPNeachpca);        
end

temp = cat(3,temp,featMean,featStd);
temp = reshape(temp,size(trainMap,1)*size(trainMap,2),size(temp,3));


polNormData = temp(mask(:)==1,:);
for i = 1:size(polNormData,2)
    polNormData(:,i) = mat2gray(polNormData(:,i));
end
PolFeat = zeros(size(temp,1),size(polNormData,2));
PolFeat(mask(:)==1,:) = polNormData;



%% label and unlabel data organization
% labeled data
labeledHSIData = HSIFeat(trainMap(:)>0,:);
labeledPOLData = PolFeat(trainMap(:)>0,:);

% unlabeled data
unLabeledData = cat(2,HSIFeat,PolFeat);
unLabeledData((mask(:)==0)|(trainMap(:)>0),:) = [];
load('.\data\LCLU_Berlin\randomUnLabeledDataIndex.mat')

[~,idx] = sort(randomIdx);
nb_unlabeldata = 6000;

unLabeledData = unLabeledData(idx(1:nb_unlabeldata),:);

unLabeledHSIData = unLabeledData(:,1:size(HSIFeat,2));
unLabeledPOLData = unLabeledData(:,size(HSIFeat,2)+1:end);



%% construct supervised part for the graph
k_ls8 = size(unLabeledHSIData,1);
k_se1 = size(unLabeledPOLData,1);
temp = trainMap(trainMap(:)>0);
temp = cat(1,temp,zeros(k_ls8,1));
S = repmat(temp,1,length(temp))==repmat(temp',length(temp'),1);
S(temp==0,:) = 0;
figure,imshow(S);
D = ~S;
D(temp==0,:) = 0;
D(:,temp==0) = 0;

%% classification 
LS8Data = cat(1,labeledHSIData,unLabeledHSIData);
SE1Data = cat(1,labeledPOLData,unLabeledPOLData);

Z = cat(2,LS8Data,SE1Data);
data = cat(2,HSIFeat,PolFeat);

k = 10:10:120;
dim = 5:5:50;

oa = zeros(length(k),length(dim),5);

Xunlab=[LS8Data, SE1Data];%


for cv_k = 1:length(k)
    K_neighbor = k(cv_k);

    map1=GGF_SE(Xunlab,LS8Data,SE1Data,dim(end),K_neighbor,S);
    
    %% classification
    oa_knn  = zeros(1,length(dim));
    oa_lsvm = zeros(1,length(dim));
    oa_ksvm = zeros(1,length(dim));
    oa_rf   = zeros(1,length(dim));
    oa_ccf  = zeros(1,length(dim));
    
    parfor cv_dim = 1:length(dim)
        feat = data * map1(:,1:dim(cv_dim));
        disp('')
        disp('---------------------')     
        
        for i = 1:size(feat,2)
            feat(:,i) = mat2gray(feat(:,i));
        end        
        
        
        trainFeat = feat(trainMap(:)>0,:);
        trainLab = trainMap(trainMap(:)>0);            

        testFeat = feat(testMap(:)>0,:);
        testLab = testMap(testMap(:)>0);

        % knn
        rng(1)
        mdl =ClassificationKNN.fit(trainFeat,trainLab,'NumNeighbors',1,'distance','euclidean'); 
        predLab= predict(mdl,testFeat); 
        [ ~,oa_knn(1,cv_dim),~,~,~ ] = confusionMatrix(testLab,predLab);


        % linear - svm 
        rng(1)
        t = templateSVM('Standardize',0,'Solver','SMO','KernelFunction','linear','KernelScale','auto');
        L_SVM1 = fitcecoc(trainFeat,trainLab,'Learners',t);
        predLab = predict(L_SVM1,testFeat);
        [ ~,oa_lsvm(1,cv_dim),~,~,~ ] = confusionMatrix(testLab,predLab);


        % rbf - svm
        rng(1)
        folds = 5;
        [ cv_C, cv_G, ~ ] = kenelPar( trainLab,trainFeat,folds );% optimal parameters searching
        K_SVM1 = svmtrain(trainLab,trainFeat,sprintf('-c %f -g %f', cv_C, cv_G));% SVM training
        [predLab] = svmpredict(testLab, testFeat, K_SVM1);% predict using trained model
        [ ~,oa_ksvm(1,cv_dim),~,~,~  ] = confusionMatrix( testLab, predLab );


        % rf
        rng(1); % For reproducibility
        NumTrees = 40;
        Mdl_rf = TreeBagger(NumTrees,trainFeat,trainLab);%,'OOBPredictorImportance','on');
        predLab = predict(Mdl_rf,testFeat);
        predLab = cellfun(@str2double,predLab);
        [ ~,oa_rf(1,cv_dim),~,~,~ ] = confusionMatrix(testLab,predLab);


        % ccf
        rng(1); % For reproducibility
        nb_trees = 40;
        [ccfs] = genCCF(nb_trees,trainFeat,trainLab);
        [predLab, ~, ~] = predictFromCCF(ccfs,testFeat);
        [ ~,oa_ccf(1,cv_dim),~,~,~ ] = confusionMatrix(testLab,predLab);
        
        
        disp(['KNN  OA:',num2str(oa_knn(1,cv_dim))])
        disp(['LSVM OA:',num2str(oa_lsvm(1,cv_dim))])
        disp(['KSVM OA:',num2str(oa_ksvm(1,cv_dim))])        
        disp(['RF   OA:',num2str(oa_rf(1,cv_dim))])
        disp(['CCF  OA:',num2str(oa_ccf(1,cv_dim))]) 
        disp(['k :',num2str(k(cv_k))])
        disp(['Dim :',num2str(dim(cv_dim))])
        disp('---------------------')
        disp('')
        
        
        
    end
    
    oa(cv_k,:,1) = oa_knn;
    oa(cv_k,:,2) = oa_lsvm;
    oa(cv_k,:,3) = oa_ksvm;
    oa(cv_k,:,4) = oa_rf;
    oa(cv_k,:,5) = oa_ccf;
    
    
end



save GGF_SE oa k dim




