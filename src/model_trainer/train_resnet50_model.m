% Train_ASL_Model.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Create Output Directory
fprintf('Creating output directory...\n');

outputDir = fullfile(pwd, 'outputs', 'resnet50_', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
fprintf('Output directory created: %s\n', outputDir);

%% SECTION 2: Targeted Data Loading
fprintf('Loading data...\n');

datasetPath = fullfile(pwd, 'data/datasets_v2/'); 

if ~exist(datasetPath, 'dir')
    fprintf('Warning: folder not found at: %s\n', datasetPath);
    fprintf('Scanning current directory...\n');
    datasetPath = pwd; 
else
    fprintf('Target folder found: %s\n', datasetPath);
end

fprintf('Scanning images...\n');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 

fprintf('Found images: %d\n', length(imds.Files));

%% SECTION 3: Filter & Split Data (STRICT A-Z ONLY)
fprintf('Filtering data (A-Z only)...\n');

labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

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
fprintf('Data split completed:\n');
fprintf(' - Training:   %d\n', length(imdsTrain.Files));
fprintf(' - Validation: %d\n', length(imdsValidation.Files));

%% SECTION 4: Load Pre-trained Network (ResNet-50)
fprintf('Loading ResNet-50...\n');
try
    net = resnet50;
    analyzeNetwork(resnet50);
    fprintf('ResNet-50 loaded.\n');
catch
    error('ResNet-50 not found. Please install the support package.');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% SECTION 5: Modify Network Layers
fprintf('Modifying layers...\n');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('Target classes: %d\n', numClasses);

if numClasses < 2
    error('Error: Fewer than 2 classes found.');
end

layersToRemove = {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'};
lgraph = removeLayers(lgraph, layersToRemove);

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');
fprintf('New layers attached.\n');

%% SECTION 6: Data Augmentation
fprintf('Configuring data augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...  
    'RandYTranslation', [-30 30], ...  
    'RandRotation', [-15 15], ...      
    'RandScale', [0.9 1.1]);           

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
fprintf('Augmentation ready.\n');

%% SECTION 7: Training Options
fprintf('Setting training options...\n');
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

fprintf('Options set.\n');

%% SECTION 8: Training Execution
fprintf('Starting training...\n');

trainingTimer = tic;

try
    [trainedNet, trainingInfo] = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, 'Error during training: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('Training complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 9: Generate and Save Confusion Matrix
fprintf('Generating confusion matrix...\n');

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
fprintf('Confusion matrix saved.\n');

%% SECTION 10: Save Trained Network
fprintf('Saving network...\n');

save(fullfile(outputDir, 'resnet50.mat'), 'trainedNet');
fprintf('Network saved.\n');