% Train_ASL_Model_NASNet.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Create Output Directory
fprintf('[Step 1] Creating Output Directory...\n');

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
outputDir = fullfile(pwd, 'outputs', timestamp);

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('   > Output directory: %s\n', outputDir);

%% SECTION 2: Targeted Data Loading
fprintf('\n[Step 2] Initializing Data Loading...\n');

datasetPath = fullfile(pwd, 'data', 'datasets'); 

if ~exist(datasetPath, 'dir')
    fprintf('   > Warning: "raw" folder not found, trying "preprocessed"...\n');
    datasetPath = fullfile(pwd, 'data', 'preprocessed');
end

if ~exist(datasetPath, 'dir')
    error('No valid data folder found.');
else
    fprintf('   > Target folder found: %s\n', datasetPath);
end

fprintf('   > Scanning for images...\n');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 

fprintf('   > Total images found: %d\n', length(imds.Files));

%% SECTION 3: Filter & Split Data (STRICT A-Z ONLY)
fprintf('\n[Step 3] Filtering Data (A-Z Only)...\n');

labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

hasEnoughData = labelCounts.Count > 50;
isLetterAZ = arrayfun(@(x) ~isempty(regexp(char(x), '^[A-Z]$', 'once')), allLabels);
validLabels = allLabels(hasEnoughData & isLetterAZ);

removedLabels = allLabels(~(hasEnoughData & isLetterAZ));
if ~isempty(removedLabels)
    fprintf('   > Removing non-alphabet classes:\n');
    disp(removedLabels');
end

fprintf('   > Keeping ONLY A-Z classes. (Valid Classes: %d)\n', length(validLabels));

filesToKeep = ismember(imds.Labels, validLabels);
imds = subset(imds, filesToKeep);
imds.Labels = removecats(imds.Labels);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('   > Training Images:   %d\n', length(imdsTrain.Files));
fprintf('   > Validation Images: %d\n', length(imdsValidation.Files));

%% SECTION 4: Preview Sample Images
fprintf('\n[Step 4] Displaying Sample Images...\n');

numSamples = 100;
gridRows = 10;
gridCols = 10;

numSamples = min(numSamples, length(imdsTrain.Files));
sampleIndices = randperm(length(imdsTrain.Files), numSamples);

fig = figure('Name', 'Training Samples', 'Position', [50 50 1200 1000]);
for i = 1:numSamples
    img = imread(imdsTrain.Files{sampleIndices(i)});
    img = im2double(img);
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    subplot(gridRows, gridCols, i);
    imshow(img);
    title(char(imdsTrain.Labels(sampleIndices(i))), 'FontSize', 7);
end
sgtitle('Training Samples');

saveas(fig, fullfile(outputDir, 'training_samples.png'));
fprintf('   > Training samples saved to: %s\n', fullfile(outputDir, 'training_samples.png'));

fprintf('   > Displaying %d samples.\n', numSamples);
fprintf('   > Close figure to continue training.\n');
waitfor(fig);

%% SECTION 5: Load Pre-trained Network (NASNet-Large)
fprintf('\n[Step 5] Loading NASNet-Large Architecture...\n');
try
    net = nasnetlarge;
    fprintf('   > NASNet-Large loaded successfully.\n');
catch
    error('NASNet-Large not found. Install "Deep Learning Toolbox Model for NASNet-Large Network".');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% SECTION 6: Modify Network Layers
fprintf('\n[Step 6] Modifying Network Layers...\n');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('   > Target Classes for ASL: %d\n', numClasses);

if numClasses ~= 26
    warning('You have %d classes, but ASL A-Z requires 26.', numClasses);
end

% Remove final layers (NASNet-Large specific)
layersToRemove = {'predictions', 'predictions_softmax', 'ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

% Add new classification layers
newLayers = [
    fullyConnectedLayer(512, 'Name', 'new_fc1', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    batchNormalizationLayer('Name', 'new_bn')
    reluLayer('Name', 'new_relu')
    dropoutLayer(0.5, 'Name', 'new_dropout')
    fullyConnectedLayer(numClasses, 'Name', 'new_fc2', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

lgraph = addLayers(lgraph, newLayers);

% Connect to NASNet-Large's global average pooling layer
lgraph = connectLayers(lgraph, 'global_average_pooling2d_2', 'new_fc1');
fprintf('   > Network graph modified.\n');

%% SECTION 7: Unfreeze Later Layers for Fine-tuning
fprintf('\n[Step 7] Configuring Layer Freezing...\n');

layers = lgraph.Layers;
connections = lgraph.Connections;

numLayers = numel(layers);
freezeThreshold = floor(numLayers * 0.6);

for i = 1:numLayers
    if isprop(layers(i), 'WeightLearnRateFactor')
        if i <= freezeThreshold
            layers(i).WeightLearnRateFactor = 0;
            layers(i).BiasLearnRateFactor = 0;
        else
            layers(i).WeightLearnRateFactor = 1;
            layers(i).BiasLearnRateFactor = 1;
        end
    end
end

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers(i));
end
for i = 1:size(connections, 1)
    lgraph = connectLayers(lgraph, connections.Source{i}, connections.Destination{i});
end

fprintf('   > Frozen first %d layers, unfrozen remaining layers.\n', freezeThreshold);

%% SECTION 8: Data Augmentation
fprintf('\n[Step 8] Configuring Data Augmentation...\n');

augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-20 20], ...  
    'RandYTranslation', [-20 20], ...  
    'RandRotation', [-10 10], ...      
    'RandScale', [0.85 1.15], ...
    'RandXReflection', false, ...
    'RandYReflection', false);

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('   > Augmentation ready.\n');

%% SECTION 9: Training Options
fprintf('\n[Step 9] Setting Training Options...\n');

% NASNet-Large requires more memory; consider reducing batch size if needed
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 8, ...           
    'MaxEpochs', 25, ...               
    'InitialLearnRate', 0.001, ...     
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.0001, ...
    'Momentum', 0.9, ...
    'Shuffle', 'every-epoch', ...      
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...              
    'Plots', 'training-progress');     

fprintf('   > Options set.\n');

%% SECTION 10: Training Execution
fprintf('\n[Step 10] Starting Training...\n');

trainingTimer = tic;

try
    trainedNet = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, '\nERROR: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('\n   > Training Complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 11: Generate Confusion Matrix
fprintf('\n[Step 11] Generating Confusion Matrix...\n');

YPred = classify(trainedNet, auimdsValidation);
YTrue = imdsValidation.Labels;

accuracy = mean(YPred == YTrue);
fprintf('   > Validation Accuracy: %.2f%%\n', accuracy * 100);

confFig = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
confMat = confusionmat(YTrue, YPred);
confusionchart(confMat, categories(YTrue), ...
    'Title', sprintf('Confusion Matrix (Accuracy: %.2f%%)', accuracy * 100), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

saveas(confFig, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('   > Confusion matrix saved to: %s\n', fullfile(outputDir, 'confusion_matrix.png'));

save(fullfile(outputDir, 'confusion_matrix_data.mat'), 'confMat', 'YPred', 'YTrue', 'accuracy');
fprintf('   > Confusion matrix data saved to: %s\n', fullfile(outputDir, 'confusion_matrix_data.mat'));

%% SECTION 12: Per-Class Accuracy Analysis
fprintf('\n[Step 12] Per-Class Accuracy Analysis...\n');

classNames = categories(YTrue);
perClassAccuracy = zeros(numel(classNames), 1);
perClassCount = zeros(numel(classNames), 1);

fprintf('\n   %-8s %-12s %-8s\n', 'Class', 'Accuracy', 'Samples');
fprintf('   %s\n', repmat('-', 1, 30));

for i = 1:numel(classNames)
    idx = YTrue == classNames{i};
    perClassCount(i) = sum(idx);
    if perClassCount(i) > 0
        perClassAccuracy(i) = mean(YPred(idx) == YTrue(idx)) * 100;
    else
        perClassAccuracy(i) = 0;
    end
    fprintf('   %-8s %-12.2f %-8d\n', classNames{i}, perClassAccuracy(i), perClassCount(i));
end

threshold = 80;
problematicIdx = perClassAccuracy < threshold;
if any(problematicIdx)
    fprintf('\n   > Classes below %.0f%% accuracy:\n', threshold);
    problematicClasses = classNames(problematicIdx);
    for i = 1:numel(problematicClasses)
        fprintf('      - %s (%.2f%%)\n', problematicClasses{i}, perClassAccuracy(strcmp(classNames, problematicClasses{i})));
    end
end

perClassResults = table(classNames, perClassAccuracy, perClassCount, ...
    'VariableNames', {'Class', 'Accuracy', 'SampleCount'});
writetable(perClassResults, fullfile(outputDir, 'per_class_accuracy.csv'));
fprintf('   > Per-class accuracy saved to: %s\n', fullfile(outputDir, 'per_class_accuracy.csv'));

%% SECTION 13: Confusion Analysis for Similar Letters
fprintf('\n[Step 13] Analysing Commonly Confused Letters...\n');

confMatNorm = confMat ./ sum(confMat, 2);
confMatNorm(isnan(confMatNorm)) = 0;

confMatOffDiag = confMatNorm;
confMatOffDiag(logical(eye(size(confMatOffDiag)))) = 0;

[sortedConfusions, sortIdx] = sort(confMatOffDiag(:), 'descend');
topN = 10;

fprintf('\n   Top %d Confusions:\n', topN);
fprintf('   %-8s %-8s %-12s\n', 'True', 'Pred', 'Rate');
fprintf('   %s\n', repmat('-', 1, 30));

for i = 1:topN
    if sortedConfusions(i) > 0
        [row, col] = ind2sub(size(confMatNorm), sortIdx(i));
        fprintf('   %-8s %-8s %-12.2f%%\n', classNames{row}, classNames{col}, sortedConfusions(i) * 100);
    end
end

%% SECTION 14: Save Model and Training Info
fprintf('\n[Step 14] Saving Model...\n');

modelFilename = fullfile(outputDir, 'ASL_Trained_Network.mat');
save(modelFilename, 'trainedNet');
fprintf('   > Network saved to: %s\n', modelFilename);

trainingInfo.timestamp = timestamp;
trainingInfo.trainingTime = trainingTime;
trainingInfo.accuracy = accuracy;
trainingInfo.numTrainingImages = length(imdsTrain.Files);
trainingInfo.numValidationImages = length(imdsValidation.Files);
trainingInfo.numClasses = numClasses;
trainingInfo.classNames = classNames;
trainingInfo.perClassAccuracy = perClassAccuracy;
trainingInfo.architecture = 'NASNet-Large';
trainingInfo.freezeThreshold = freezeThreshold;

save(fullfile(outputDir, 'training_info.mat'), 'trainingInfo');
fprintf('   > Training info saved to: %s\n', fullfile(outputDir, 'training_info.mat'));

fprintf('\n=== SUCCESS ===\n');
fprintf('All outputs saved to: %s\n', outputDir);