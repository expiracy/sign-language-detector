clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Data Loading
fprintf('[Step 1] Loading Data...\n');

datasetPath = fullfile(pwd, 'data/');
if ~exist(datasetPath, 'dir')
    datasetPath = pwd;
end

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @readAndNormaliseImage);

fprintf(' > Total images: %d\n', numel(imds.Files));

testImg = readimage(imds, 1);
fprintf(' > Test image size: %s\n', mat2str(size(testImg)));
fprintf(' > Test image range: [%.3f, %.3f]\n', min(testImg(:)), max(testImg(:)));

%% SECTION 2: Filter to A-Z Only
fprintf('\n[Step 2] Filtering Data...\n');

labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

hasEnoughData = labelCounts.Count > 50;
isLetterAZ = arrayfun(@(x) ~isempty(regexp(char(x), '^[A-Z]$', 'once')), allLabels);
validLabels = allLabels(hasEnoughData & isLetterAZ);

filesToKeep = ismember(imds.Labels, validLabels);
imds = subset(imds, filesToKeep);
imds.Labels = removecats(imds.Labels);

numClasses = numel(categories(imds.Labels));
fprintf(' > Valid classes: %d\n', numClasses);

%% SECTION 3: Split Data
fprintf('\n[Step 3] Splitting Data...\n');

[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.7, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, 0.5, 'randomized');

fprintf(' > Training: %d\n', numel(imdsTrain.Files));
fprintf(' > Validation: %d\n', numel(imdsValidation.Files));
fprintf(' > Test: %d\n', numel(imdsTest.Files));

%% SECTION 4: Load and Modify ResNet-18
fprintf('\n[Step 4] Configuring ResNet-18...\n');

try
    net = resnet18;
    fprintf(' > ResNet-18 loaded successfully.\n');
catch
    error('ResNet-18 not found. Install "Deep Learning Toolbox Model for ResNet-18 Network".');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'new_fc');
fprintf(' > Network configured.\n');

%% SECTION 5: Data Augmentation
fprintf('\n[Step 5] Configuring Augmentation...\n');

imdsTrainAug = imageDatastore(imdsTrain.Files, ...
    'Labels', imdsTrain.Labels, ...
    'ReadFcn', @readAndAugmentImage);

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', [-30 30], ...
    'RandYTranslation', [-30 30], ...
    'RandRotation', [-15 15], ...
    'RandScale', [0.85 1.15], ...
    'RandXShear', [-10 10], ...
    'RandYShear', [-10 10]);

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrainAug, ...
    'DataAugmentation', augmenter);

auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
auimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

fprintf(' > Augmentation configured.\n');

%% SECTION 6: Visualise Augmentation
fprintf('\n[Step 6] Visualising Augmented Samples...\n');

figure('Name', 'Augmentation Examples', 'Position', [100 100 1200 600]);

for i = 1:4
    subplot(3, 4, i);
    img = readimage(imdsTrain, i);
    imshow(img);
    title(sprintf('Original: %s', char(imdsTrain.Labels(i))));
end

reset(auimdsTrain);
batch = read(auimdsTrain);
for i = 1:4
    subplot(3, 4, i + 4);
    img = batch.input{i};
    imshow(img);
    title(sprintf('Augmented: %s', char(batch.response(i))));
end

batch = read(auimdsTrain);
for i = 1:4
    subplot(3, 4, i + 8);
    img = batch.input{i};
    imshow(img);
    title(sprintf('Augmented: %s', char(batch.response(i))));
end

sgtitle('Original vs Augmented Images');
drawnow;

%% SECTION 7: Training Options
fprintf('\n[Step 7] Setting Training Options...\n');

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...
    'VerboseFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

fprintf(' > Training options configured.\n');

%% SECTION 8: Train
fprintf('\n[Step 8] Starting Training...\n');

tic;
trainedNet = trainNetwork(auimdsTrain, lgraph, options);
trainingTime = toc;
fprintf(' > Training complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 9: Evaluate
fprintf('\n[Step 9] Evaluating...\n');

predictedLabels = classify(trainedNet, auimdsTest);
trueLabels = imdsTest.Labels;

accuracy = mean(predictedLabels == trueLabels);
fprintf(' > Test Accuracy: %.2f%%\n', accuracy * 100);

figure('Name', 'Confusion Matrix', 'Position', [100 100 800 700]);
confusionchart(trueLabels, predictedLabels, ...
    'Title', sprintf('Test Accuracy: %.2f%%', accuracy * 100), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

%% SECTION 10: Save to Timestamped Output Folder
fprintf('\n[Step 10] Saving Model...\n');

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
outputFolder = fullfile(pwd, '..', 'outputs', [timestamp '_resnet18']);

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

networkPath = fullfile(outputFolder, 'ASL_ResNet18_Trained.mat');
confusionMatrixPath = fullfile(outputFolder, 'confusion_matrix.png');

save(networkPath, 'trainedNet', 'accuracy');
saveas(gcf, confusionMatrixPath);

fprintf(' > Saved to: %s\n', outputFolder);
fprintf('\n=== Complete ===\n');

%% ========== LOCAL FUNCTIONS ==========

function img = readAndNormaliseImage(filename)
    [img, cmap] = imread(filename);
    
    if ~isempty(cmap)
        img = ind2rgb(img, cmap);
    end
    
    img = im2double(img);
    
    if size(img, 3) == 4
        img = img(:,:,1:3);
    end
    
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    img = max(0, min(1, img));
end

function img = readAndAugmentImage(filename)
    [img, cmap] = imread(filename);
    
    if ~isempty(cmap)
        img = ind2rgb(img, cmap);
    end
    
    img = im2double(img);
    
    if size(img, 3) == 4
        img = img(:,:,1:3);
    end
    
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    img = max(0, min(1, img));
    
    if rand < 0.7
        brightnessOffset = (rand - 0.5) * 0.3;
        img = img + brightnessOffset;
    end
    
    if rand < 0.7
        contrastFactor = 0.75 + rand * 0.5;
        img = (img - 0.5) * contrastFactor + 0.5;
    end
    
    if rand < 0.5
        img = max(0, min(1, img));
        hsvImg = rgb2hsv(img);
        saturationFactor = 0.7 + rand * 0.6;
        hsvImg(:,:,2) = min(1, hsvImg(:,:,2) * saturationFactor);
        img = hsv2rgb(hsvImg);
    end
    
    if rand < 0.3
        noiseLevel = rand * 0.03;
        img = img + noiseLevel * randn(size(img));
    end
    
    if rand < 0.4
        img = max(0, min(1, img));
        gamma = 0.8 + rand * 0.4;
        img = img .^ gamma;
    end
    
    img = max(0, min(1, img));
end