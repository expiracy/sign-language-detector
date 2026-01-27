% Train_ASL_Model.m

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

datasetPath = fullfile(pwd, 'data', 'preprocessed'); 

if ~exist(datasetPath, 'dir')
    fprintf('   > Warning: "DataSets" folder not found at: %s\n', datasetPath);
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

% Save the training samples figure
saveas(fig, fullfile(outputDir, 'training_samples.png'));
fprintf('   > Training samples saved to: %s\n', fullfile(outputDir, 'training_samples.png'));

fprintf('   > Displaying %d samples.\n', numSamples);
fprintf('   > Close figure to continue training.\n');
waitfor(fig);

%% SECTION 5: Load Pre-trained Network (GoogLeNet)
fprintf('\n[Step 5] Loading GoogLeNet Architecture...\n');
try
    net = googlenet;
    fprintf('   > GoogLeNet loaded successfully.\n');
catch
    error('GoogLeNet not found. Install "Deep Learning Toolbox Model for GoogLeNet Network".');
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

layersToRemove = {'loss3-classifier', 'prob', 'output'};
lgraph = removeLayers(lgraph, layersToRemove);

newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
fprintf('   > Network graph modified.\n');

%% SECTION 7: Data Augmentation
fprintf('\n[Step 7] Configuring Data Augmentation...\n');

augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...  
    'RandYTranslation', [-30 30], ...  
    'RandRotation', [-15 15], ...      
    'RandScale', [0.9 1.1]);           

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

fprintf('   > Augmentation ready.\n');

%% SECTION 8: Training Options
fprintf('\n[Step 8] Setting Training Options...\n');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...           
    'MaxEpochs', 6, ...                
    'InitialLearnRate', 0.0001, ...    
    'Shuffle', 'every-epoch', ...      
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 50, ...     
    'Verbose', true, ...              
    'Plots', 'training-progress');     

fprintf('   > Options set.\n');

%% SECTION 9: Training Execution
fprintf('\n[Step 9] Starting Training...\n');

trainingTimer = tic;

try
    trainedNet = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, '\nERROR: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('\n   > Training Complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 10: Generate Confusion Matrix
fprintf('\n[Step 10] Generating Confusion Matrix...\n');

% Classify validation images
YPred = classify(trainedNet, auimdsValidation);
YTrue = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YTrue);
fprintf('   > Validation Accuracy: %.2f%%\n', accuracy * 100);

% Create confusion matrix figure
confFig = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
confMat = confusionmat(YTrue, YPred);
confusionchart(confMat, categories(YTrue), ...
    'Title', sprintf('Confusion Matrix (Accuracy: %.2f%%)', accuracy * 100), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Save confusion matrix figure
saveas(confFig, fullfile(outputDir, 'confusion_matrix.png'));
fprintf('   > Confusion matrix saved to: %s\n', fullfile(outputDir, 'confusion_matrix.png'));

% Save confusion matrix data
save(fullfile(outputDir, 'confusion_matrix_data.mat'), 'confMat', 'YPred', 'YTrue', 'accuracy');
fprintf('   > Confusion matrix data saved to: %s\n', fullfile(outputDir, 'confusion_matrix_data.mat'));

%% SECTION 11: Save Model and Training Info
fprintf('\n[Step 11] Saving Model...\n');

% Save the trained network
modelFilename = fullfile(outputDir, 'ASL_Trained_Network.mat');
save(modelFilename, 'trainedNet');
fprintf('   > Network saved to: %s\n', modelFilename);

% Save training information
trainingInfo.timestamp = timestamp;
trainingInfo.trainingTime = trainingTime;
trainingInfo.accuracy = accuracy;
trainingInfo.numTrainingImages = length(imdsTrain.Files);
trainingInfo.numValidationImages = length(imdsValidation.Files);
trainingInfo.numClasses = numClasses;
trainingInfo.classNames = categories(imdsTrain.Labels);

save(fullfile(outputDir, 'training_info.mat'), 'trainingInfo');
fprintf('   > Training info saved to: %s\n', fullfile(outputDir, 'training_info.mat'));

fprintf('\n=== SUCCESS ===\n');
fprintf('All outputs saved to: %s\n', outputDir);