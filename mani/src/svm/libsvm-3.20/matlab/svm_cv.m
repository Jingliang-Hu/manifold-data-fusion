function [ claMap, overallAccuracy, accuracy_cv, cv_C, cv_G, model ] = svm_cv( data,percent )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
[ label ] = labelOfSample( data );
[ trainSample,trainlabel,testSample,testlabel ] = sampleData( data,label,percent );
[ temp,H ] = trainingSet(data,6 );

% [ normalised,m,s ] = featureNormalise( trainSampleH,3 );
% testNol = (testSampleH - repmat(m,size(testSampleH,1),1))./repmat(s,size(testSampleH,1),1);
% predictNol = (temp_h - repmat(m,size(temp_h,1),1))./repmat(s,size(temp_h,1),1);

normalised = trainSample;
testNol = testSample;
predictNol = temp;

% matlab SVM
% t = templateSVM('Standardize',1,'KernelFunction','gaussian');
% SVM_HSI = fitcecoc(trainSampleH,trainlabelH,'Learners',t);

% LIBSVM  
folds = 3;

tic
[ cv_C, cv_G, accuracy_cv ] = kenelPar( trainlabel,normalised,folds );
model = svmtrain(trainlabel,normalised,sprintf('-c %f -g %f', cv_C, cv_G));
toc

tic
[classTest_HSI] = svmpredict(testlabel, testNol, model);
toc

overallAccuracy = sum(classTest_HSI==testlabel)/length(testlabel);

[HSIClass_h] = svmpredict(H, predictNol, model);
claMap = reshape(HSIClass_h,1952,[]);
figure, imshow(claMap,[]);colormap(jet);

end

