% Train_ASL_Model.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Targeted Data Loading
fprintf('[Step 1] Initializing Data Loading...\n');

datasetPath = fullfile(pwd, 'DataSets'); 

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

%% SECTION 2: Filter & Split Data (STRICT A-Z ONLY)
fprintf('\n[Step 2] Filtering Data (A-Z Only)...\n');

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

%% SECTION 3: Preview Original vs Preprocessed
fprintf('\n[Step 3] Displaying Original vs Preprocessed Samples...\n');

numSamples = 10;
sampleIndices = randperm(length(imdsTrain.Files), numSamples);

figure('Name', 'Original vs Preprocessed', 'Position', [50 50 1500 600]);
for i = 1:numSamples
    % Read original image
    originalImg = imread(imdsTrain.Files{sampleIndices(i)});
    originalImg = im2double(originalImg);
    if size(originalImg, 3) == 1
        originalImg = repmat(originalImg, [1 1 3]);
    end
    
    % Get preprocessed image
    processedImg = readAndPreprocess(imdsTrain.Files{sampleIndices(i)});
    
    label = imdsTrain.Labels(sampleIndices(i));
    
    % Original on top row
    subplot(2, numSamples, i);
    imshow(originalImg);
    title(sprintf('%s (Original)', char(label)));
    
    % Preprocessed on bottom row
    subplot(2, numSamples, i + numSamples);
    imshow(processedImg);
    title('Processed');
end
sgtitle('Original vs Preprocessed Training Samples');

fprintf('   > Displaying %d random samples. Close figure to continue.\n', numSamples);
uiwait(gcf);

%% SECTION 4: Apply Preprocessing via ReadFcn
fprintf('\n[Step 4] Configuring Preprocessing...\n');

imdsTrain.ReadFcn = @readAndPreprocess;
imdsValidation.ReadFcn = @readAndPreprocess;

fprintf('   > Preprocessing pipeline attached.\n');

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
    'Verbose', false, ...              
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

%% SECTION 10: Save Result
fprintf('\n[Step 10] Saving Model...\n');
saveFilename = 'ASL_Trained_Network.mat';
save(saveFilename, 'trainedNet');

fprintf('   > Network saved to: %s\n', fullfile(pwd, saveFilename));
fprintf('=== SUCCESS ===\n');

%% ========== PREPROCESSING FUNCTIONS ==========
function img = readAndPreprocess(filename)
    img = imread(filename);
    img = im2double(img);
    
    if size(img, 3) == 4
        img = img(:,:,1:3);
    end
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
    
    img = preprocessingPipeline(img);
end

function img = preprocessingPipeline(img)
    img = im2double(img);
    
    img = imflatfield(img, 5);
    
    if size(img, 3) == 3
        lab = rgb2lab(img);
        lab(:,:,1) = adapthisteq(lab(:,:,1) / 100, 'ClipLimit', 0.003) * 100;
        img = lab2rgb(lab);
        img = max(0, min(1, img));
    else
        img = adapthisteq(img, 'ClipLimit', 0.003);
    end
    
    img = imsharpen(img, 'Radius', 0.5, 'Amount', 0.5, 'Threshold', 0.3);
    img = im2uint8(img);
end