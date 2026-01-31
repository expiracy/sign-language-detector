% Train_ASL_Model_GoogLeNet.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Create Output Directory
fprintf('<strong>[Step 1] Creating Output Directory...</strong>\n');

outputDir = fullfile(pwd, 'outputs', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
fprintf('   > Output directory created: %s\n', outputDir);

%% SECTION 2: Targeted Data Loading
fprintf('\n<strong>[Step 2] Initialising Data Loading...</strong>\n');

datasetPath = fullfile(pwd, 'data/datasets/ASL_DATA_Test/');

if ~exist(datasetPath, 'dir')
    fprintf('   > Warning: dataset folder not found at: %s\n', datasetPath);
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
fprintf('\n<strong>[Step 3] Filtering Data (A-Z Only)...</strong>\n');

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

%% SECTION 4: Load Pre-trained Network (GoogLeNet)
fprintf('\n<strong>[Step 4] Loading GoogLeNet Architecture...</strong>\n');
try
    net = googlenet;
    fprintf('   > GoogLeNet loaded successfully.\n');
catch
    error('CRITICAL ERROR: GoogLeNet not found. Please install "Deep Learning Toolbox Model for GoogLeNet Network".');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%% SECTION 5: Modify Network Layers
fprintf('\n<strong>[Step 5] Modifying Network Layers...</strong>\n');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('   > Target Classes for ASL: %d (Must be 26)\n', numClasses);

if numClasses ~= 26
    warning('Warning: You have %d classes, but ASL A-Z requires 26. Check your folders.', numClasses);
end

if numClasses < 2
    error('CRITICAL ERROR: Found fewer than 2 classes.');
end

% Find the final learnable layer and the classification output layer
[learnableLayer, classLayer] = findLayersToReplaceLocal(lgraph);

% Replace learnable layer with correct number of classes
if isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
elseif isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    % GoogLeNet commonly uses a 1x1 conv as the classifier
    newLearnableLayer = convolution2dLayer(1, numClasses, ...
        'Name', 'new_conv', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
else
    error('Unsupported learnable layer type: %s', class(learnableLayer));
end

newClassLayer = classificationLayer('Name', 'classoutput');

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

fprintf('   > Replaced "%s" and "%s". Network graph is valid.\n', learnableLayer.Name, classLayer.Name);

%% SECTION 6: Data Augmentation
fprintf('\n<strong>[Step 6] Configuring Data Augmentation...</strong>\n');
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
fprintf('\n<strong>[Step 7] Setting Training Options...</strong>\n');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 0.0001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

fprintf('   > Options set. Visual Progress Window will launch shortly.\n');

%% SECTION 8: Training Execution
fprintf('\n<strong>[Step 8] Starting Training...</strong>\n');
fprintf('   > A separate window will open to show the Progress Bar.\n');

trainingTimer = tic;

try
    [trainedNet, trainingInfo] = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, '\nERROR DURING TRAINING: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer);
fprintf('\n   > Training Complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 9: Save Training Progress Plot
fprintf('\n<strong>[Step 9] Saving Training Progress Plot...</strong>\n');

figTraining = figure('Name', 'Training Progress', 'Position', [100 100 1200 500]);

subplot(1,2,1);
plot(trainingInfo.TrainingLoss, 'b-', 'LineWidth', 1.5);
hold on;
valLossIdx = ~isnan(trainingInfo.ValidationLoss);
valIterations = find(valLossIdx);
plot(valIterations, trainingInfo.ValidationLoss(valLossIdx), 'r-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Loss');
title('Training and Validation Loss');
legend('Training Loss', 'Validation Loss', 'Location', 'northeast');
grid on;
hold off;

subplot(1,2,2);
plot(trainingInfo.TrainingAccuracy, 'b-', 'LineWidth', 1.5);
hold on;
valAccIdx = ~isnan(trainingInfo.ValidationAccuracy);
valIterations = find(valAccIdx);
plot(valIterations, trainingInfo.ValidationAccuracy(valAccIdx), 'r-', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Accuracy (%)');
title('Training and Validation Accuracy');
legend('Training Accuracy', 'Validation Accuracy', 'Location', 'southeast');
grid on;
hold off;

saveas(figTraining, fullfile(outputDir, 'training_progress.png'));
saveas(figTraining, fullfile(outputDir, 'training_progress.fig'));
fprintf('   > Training progress plot saved.\n');

%% SECTION 10: Generate and Save Confusion Matrix
fprintf('\n<strong>[Step 10] Generating Confusion Matrix...</strong>\n');

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
saveas(figConfusion, fullfile(outputDir, 'confusion_matrix.fig'));
fprintf('   > Confusion matrix saved.\n');

%% SECTION 11: Calculate and Save Per-Class Metrics
fprintf('\n<strong>[Step 11] Calculating Per-Class Metrics...</strong>\n');

confMat = confusionmat(trueLabels, predictedLabels);
classNames = categories(trueLabels);
numClassesActual = numel(classNames);

precision = zeros(numClassesActual, 1);
recall = zeros(numClassesActual, 1);
f1Score = zeros(numClassesActual, 1);

for i = 1:numClassesActual
    tp = confMat(i, i);
    fp = sum(confMat(:, i)) - tp;
    fn = sum(confMat(i, :)) - tp;

    if (tp + fp) > 0
        precision(i) = tp / (tp + fp);
    end

    if (tp + fn) > 0
        recall(i) = tp / (tp + fn);
    end

    if (precision(i) + recall(i)) > 0
        f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
end

metricsTable = table(classNames, precision * 100, recall * 100, f1Score * 100, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});

figMetrics = figure('Name', 'Per-Class Metrics', 'Position', [100 100 1000 600]);
bar([precision, recall, f1Score] * 100);
set(gca, 'XTickLabel', classNames);
xlabel('Class');
ylabel('Percentage (%)');
title('Per-Class Performance Metrics');
legend('Precision', 'Recall', 'F1 Score', 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

saveas(figMetrics, fullfile(outputDir, 'per_class_metrics.png'));
saveas(figMetrics, fullfile(outputDir, 'per_class_metrics.fig'));
fprintf('   > Per-class metrics plot saved.\n');

%% SECTION 12: Save All Results
fprintf('\n<strong>[Step 12] Saving All Results...</strong>\n');

save(fullfile(outputDir, 'trained_network.mat'), 'trainedNet');
fprintf('   > Trained network saved.\n');

save(fullfile(outputDir, 'training_info.mat'), 'trainingInfo');
fprintf('   > Training info saved.\n');

writetable(metricsTable, fullfile(outputDir, 'per_class_metrics.csv'));
fprintf('   > Per-class metrics CSV saved.\n');

summary.NetworkArchitecture = 'GoogLeNet';
summary.TrainingDate = datestr(now);
summary.TrainingTimeMinutes = trainingTime / 60;
summary.TotalEpochs = 20;
summary.MiniBatchSize = 32;
summary.InitialLearningRate = 0.0001;
summary.TrainingImages = length(imdsTrain.Files);
summary.ValidationImages = length(imdsValidation.Files);
summary.NumClasses = numClasses;
summary.FinalValidationAccuracy = validationAccuracy;
summary.FinalTrainingLoss = trainingInfo.TrainingLoss(end);
summary.MeanPrecision = mean(precision) * 100;
summary.MeanRecall = mean(recall) * 100;
summary.MeanF1Score = mean(f1Score) * 100;

save(fullfile(outputDir, 'training_summary.mat'), 'summary');

summaryFileID = fopen(fullfile(outputDir, 'training_summary.txt'), 'w');
fprintf(summaryFileID, 'ASL Model Training Summary\n');
fprintf(summaryFileID, '==========================\n\n');
fprintf(summaryFileID, 'Network Architecture: %s\n', summary.NetworkArchitecture);
fprintf(summaryFileID, 'Training Date: %s\n', summary.TrainingDate);
fprintf(summaryFileID, 'Training Time: %.2f minutes\n\n', summary.TrainingTimeMinutes);
fprintf(summaryFileID, 'Training Parameters:\n');
fprintf(summaryFileID, '  - Epochs: %d\n', summary.TotalEpochs);
fprintf(summaryFileID, '  - Mini-Batch Size: %d\n', summary.MiniBatchSize);
fprintf(summaryFileID, '  - Initial Learning Rate: %.4f\n\n', summary.InitialLearningRate);
fprintf(summaryFileID, 'Dataset:\n');
fprintf(summaryFileID, '  - Training Images: %d\n', summary.TrainingImages);
fprintf(summaryFileID, '  - Validation Images: %d\n', summary.ValidationImages);
fprintf(summaryFileID, '  - Number of Classes: %d\n\n', summary.NumClasses);
fprintf(summaryFileID, 'Results:\n');
fprintf(summaryFileID, '  - Final Validation Accuracy: %.2f%%\n', summary.FinalValidationAccuracy);
fprintf(summaryFileID, '  - Final Training Loss: %.4f\n', summary.FinalTrainingLoss);
fprintf(summaryFileID, '  - Mean Precision: %.2f%%\n', summary.MeanPrecision);
fprintf(summaryFileID, '  - Mean Recall: %.2f%%\n', summary.MeanRecall);
fprintf(summaryFileID, '  - Mean F1 Score: %.2f%%\n', summary.MeanF1Score);
fclose(summaryFileID);
fprintf('   > Training summary text file saved.\n');

save(fullfile(outputDir, 'confusion_matrix_data.mat'), ...
    'confMat', 'classNames', 'predictedLabels', 'trueLabels');
fprintf('   > Confusion matrix data saved.\n');

fprintf('\n<strong>=== SUCCESS: All outputs saved to %s ===</strong>\n', outputDir);

%% ------------------------------------------------------------------------
%% Local helper: find final learnable + classification layers (robust)
function [learnableLayer, classLayer] = findLayersToReplaceLocal(lgraph)

layers = lgraph.Layers;
connections = lgraph.Connections;

% 1) Find the classification output layer (from the end)
classLayer = [];
for i = numel(layers):-1:1
    if isa(layers(i), 'nnet.cnn.layer.ClassificationOutputLayer')
        classLayer = layers(i);
        break;
    end
end
if isempty(classLayer)
    error('No classification output layer found.');
end

% 2) Walk backwards until we hit a learnable layer (FC or Conv2D)
currentName = classLayer.Name;

while true
    idx = strcmp(connections.Destination, currentName);
    if ~any(idx)
        error('Could not trace connections back from layer "%s".', currentName);
    end

    prevName = connections.Source{find(idx, 1, 'first')};
    prevLayer = layers(strcmp({layers.Name}, prevName));

    if isa(prevLayer, 'nnet.cnn.layer.FullyConnectedLayer') || isa(prevLayer, 'nnet.cnn.layer.Convolution2DLayer')
        learnableLayer = prevLayer;
        return;
    end

    currentName = prevName;
end

end
