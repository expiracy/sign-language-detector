% Train_ASL_Model_ResNet18.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% Create Output Directory
outputDir = fullfile(pwd, 'outputs', sprintf('resnet18_%s', datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Data Loading
datasetPath = fullfile(pwd, 'data/datasets_v2/'); 

if ~exist(datasetPath, 'dir')
    fprintf('Warning: folder not found at: %s\n', datasetPath);
    datasetPath = pwd; 
end

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 

fprintf('Found %d images\n', length(imds.Files));

%% Filter & Split Data
labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

% Only keeping folders named A-Z. Ignoring "nothing", "del", "space", etc.
hasEnoughData = labelCounts.Count > 50;
isLetterAZ = arrayfun(@(x) ~isempty(regexp(char(x), '^[A-Z]$', 'once')), allLabels);

validLabels = allLabels(hasEnoughData & isLetterAZ);

removedLabels = allLabels(~(hasEnoughData & isLetterAZ));
if ~isempty(removedLabels)
    fprintf('Removing non-alphabet classes:\n');
    disp(removedLabels');
end

fprintf('Keeping A-Z classes (Valid: %d)\n', length(validLabels));

filesToKeep = ismember(imds.Labels, validLabels);
imds = subset(imds, filesToKeep);

imds.Labels = removecats(imds.Labels);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('Training: %d | Validation: %d\n', length(imdsTrain.Files), length(imdsValidation.Files));

%% Load ResNet-18
try
    net = resnet18;
catch
    error('ResNet-18 not found. Please install the support package.');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% Modify Network Layers
numClasses = numel(categories(imdsTrain.Labels));

if numClasses < 2
    error('Error: Fewer than 2 classes found.');
end

layersToRemove = {'fc1000', 'prob', 'ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'new_fc');

%% Data Augmentation
augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...  
    'RandYTranslation', [-30 30], ...  
    'RandRotation', [-15 15], ...      
    'RandScale', [0.9 1.1]);           

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% Training Options
epochs = 5;

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', epochs, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


%% Training
trainingTimer = tic;

try
    [trainedNet, trainingInfo] = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, 'Error during training: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('Training complete in %.2f minutes.\n', trainingTime/60);

%% Confusion Matrix
predictedLabels = classify(trainedNet, auimdsValidation);
trueLabels = imdsValidation.Labels;

validationAccuracy = mean(predictedLabels == trueLabels) * 100;
fprintf('Validation Accuracy: %.2f%%\n', validationAccuracy);

figConfusion = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
confusionchart(trueLabels, predictedLabels, ...
    'Title', sprintf('Confusion Matrix (Accuracy: %.2f%%)', validationAccuracy), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

saveas(figConfusion, fullfile(outputDir, 'confusion_matrix.png'));

%% Save Network
save(fullfile(outputDir, 'resnet18.mat'), 'trainedNet');