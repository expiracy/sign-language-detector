% Train_ASL_Model_ResNet101.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Create Output Directory
fprintf('[Step 1] Creating Output Directory...\n');

outputDir = fullfile(pwd, 'outputs', 'resnet101_', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
fprintf('   > Output directory created: %s\n', outputDir);

%% SECTION 2: Targeted Data Loading
fprintf('[Step 2] Initializing Data Loading...\n');

datasetPath = fullfile(pwd, 'data/datasets_v2/'); 

if ~exist(datasetPath, 'dir')
    fprintf('   > Warning: folder not found at: %s\n', datasetPath);
    fprintf('   > Scanning current directory instead...\n');
    datasetPath = pwd; 
else
    fprintf('   > Target folder found: %s\n', datasetPath);
end

fprintf('   > Scanning for images...\n');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 

fprintf('   > Total images found: %d\n', length(imds.Files));

%% SECTION 3: Filter & Split Data (STRICT A-Z ONLY)
fprintf('[Step 3] Filtering Data (A-Z Only)...\n');

labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

hasEnoughData = labelCounts.Count > 50;
isLetterAZ = arrayfun(@(x) ~isempty(regexp(char(x), '^[A-Z]$', 'once')), allLabels);

validLabels = allLabels(hasEnoughData & isLetterAZ);

removedLabels = allLabels(~(hasEnoughData & isLetterAZ));
if ~isempty(removedLabels)
    fprintf('   > REMOVING the following non-alphabet classes:\n');
    disp(removedLabels');
end

fprintf('   > Keeping ONLY A-Z classes. (Valid Classes: %d)\n', length(validLabels));

filesToKeep = ismember(imds.Labels, validLabels);
imds = subset(imds, filesToKeep);

imds.Labels = removecats(imds.Labels);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('   > Data Split Completed:\n');
fprintf('     - Training Images:   %d\n', length(imdsTrain.Files));
fprintf('     - Validation Images: %d\n', length(imdsValidation.Files));

%% SECTION 4: Load Pre-trained Network (ResNet-101)
fprintf('[Step 4] Loading ResNet-101 Architecture...\n');
try
    net = resnet101;
    fprintf('   > ResNet-101 loaded successfully.\n');
catch
    error('CRITICAL ERROR: ResNet-101 not found. Please install "Deep Learning Toolbox Model for ResNet-101 Network".');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% SECTION 5: Modify Network Layers
fprintf('[Step 5] Modifying Network Layers...\n');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('   > Target Classes for ASL: %d\n', numClasses);

if numClasses < 2
    error('CRITICAL ERROR: Found fewer than 2 classes.');
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
fprintf('   > New layers attached. Network graph is valid.\n');

%% SECTION 6: Data Augmentation
fprintf('[Step 6] Configuring Data Augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...  
    'RandYTranslation', [-30 30], ...  
    'RandRotation', [-15 15], ...      
    'RandScale', [0.9 1.1]);           

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
fprintf('   > Augmentation ready.\n');

%% SECTION 7: Training Options
fprintf('[Step 7] Setting Training Options...\n');
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

fprintf('   > Options set. Visual Progress Window will launch shortly.\n');

%% SECTION 8: Training Execution
fprintf('[Step 8] Starting Training...\n');

trainingTimer = tic;

try
    [trainedNet, trainingInfo] = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, 'ERROR DURING TRAINING: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('   > Training Complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 9: Generate and Save Confusion Matrix
fprintf('[Step 9] Generating Confusion Matrix...\n');

predictedLabels = classify(trainedNet, auimdsValidation);
trueLabels = imdsValidation.Labels;

validationAccuracy = mean(predictedLabels == trueLabels) * 100;
fprintf('   > Validation Accuracy: %.2f%%\n', validationAccuracy);

figConfusion = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
confusionchart(trueLabels, predictedLabels, ...
    'Title', sprintf('Confusion Matrix (Accuracy: %.2f%%)', validationAccuracy), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

saveas(figConfusion, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('   > Confusion matrix saved.\n');

%% SECTION 10: Save Trained Network
fprintf('[Step 10] Saving Trained Network...\n');

save(fullfile(outputDir, 'resnet101.mat'), 'trainedNet');
fprintf('   > Trained network saved.\n');