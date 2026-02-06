% Train_ASL_Model.m

clc; clear; close all;

parallel.gpu.enableCUDAForwardCompatibility(true);

%% SECTION 1: Targeted Data Loading
fprintf('[Step 1] Initializing Data Loading...\n');

% Define the specific path
datasetPath = fullfile(pwd, 'DataSets'); 

if ~exist(datasetPath, 'dir')
    fprintf('   > Warning: "DataSets" folder not found at: %s\n', datasetPath);
    fprintf('   > Scanning current directory instead...\n');
    datasetPath = pwd; 
else
    fprintf('   > Target folder found: %s\n', datasetPath);
end

% Create the ImageDatastore
fprintf('   > Scanning for images...\n');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 

fprintf('   > Total images found: %d\n', length(imds.Files));

%% SECTION 2: Filter & Split Data (STRICT A-Z ONLY)
fprintf('\n[Step 2] Filtering Data (A-Z Only)...\n');

% Count images per class
labelCounts = countEachLabel(imds);
allLabels = labelCounts.Label;

% RULE 1: Must have enough data (> 50 images)
hasEnoughData = labelCounts.Count > 50;

% RULE 2: STRICT FILTER for Single Letters A-Z
% We use Regular Expressions to check if the folder name is exactly one letter A-Z.
isLetterAZ = arrayfun(@(x) ~isempty(regexp(char(x), '^[A-Z]$', 'once')), allLabels);

% Combine rules
validLabels = allLabels(hasEnoughData & isLetterAZ);

% Display what is being removed for debugging
removedLabels = allLabels(~(hasEnoughData & isLetterAZ));
if ~isempty(removedLabels)
    fprintf('   > REMOVING the following non-alphabet classes:\n');
    disp(removedLabels');
end

fprintf('   > Keeping ONLY A-Z classes. (Valid Classes: %d)\n', length(validLabels));

% Update the datastore to include ONLY the valid files
filesToKeep = ismember(imds.Labels, validLabels);
imds = subset(imds, filesToKeep);


imds.Labels = removecats(imds.Labels);

% SPLIT: 80% for Training, 20% for Validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('   > Data Split Completed:\n');
fprintf('     - Training Images:   %d\n', length(imdsTrain.Files));
fprintf('     - Validation Images: %d\n', length(imdsValidation.Files));

%% SECTION 3: Load Pre-trained Network (GoogLeNet)
fprintf('\n[Step 3] Loading GoogLeNet Architecture...\n');
try
    net = googlenet;
    fprintf('   > GoogLeNet loaded successfully.\n');
catch
    error('CRITICAL ERROR: GoogLeNet not found. Please install "Deep Learning Toolbox Model for GoogLeNet Network".');
end

lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize; % Usually 224x224x3

%% SECTION 4: Modify Network Layers
fprintf('\n[Step 4] Modifying Network Layers...\n');

% Now this will correctly return 26
numClasses = numel(categories(imdsTrain.Labels));
fprintf('   > Target Classes for ASL: %d (Must be 26)\n', numClasses);

if numClasses ~= 26
    warning('Warning: You have %d classes, but ASL A-Z requires 26. Check your folders.', numClasses);
end

if numClasses < 2
    error('CRITICAL ERROR: Found fewer than 2 classes.');
end

% REMOVAL: Remove the old classification head
layersToRemove = {'loss3-classifier', 'prob', 'output'};
lgraph = removeLayers(lgraph, layersToRemove);

% ADDITION: Create new head for 26 classes
newLayers = [
    fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

% CONNECT
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
fprintf('   > New layers attached. Network graph is valid.\n');

%% SECTION 5: Data Augmentation
fprintf('\n[Step 5] Configuring Data Augmentation...\n');
augmenter = imageDataAugmenter( ...
    'RandXTranslation', [-30 30], ...  
    'RandYTranslation', [-30 30], ...  
    'RandRotation', [-15 15], ...      
    'RandScale', [0.9 1.1]);           

auimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
auimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
fprintf('   > Augmentation ready.\n');

%% SECTION 6: Training Options
fprintf('\n[Step 6] Setting Training Options...\n');

% 'Plots', 'training-progress' opens the external window with the progress bar.
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...           
    'MaxEpochs', 6, ...                
    'InitialLearnRate', 0.0001, ...    
    'Shuffle', 'every-epoch', ...      
    'ValidationData', auimdsValidation, ...
    'ValidationFrequency', 50, ...     
    'Verbose', false, ...              
    'Plots', 'training-progress');     

fprintf('   > Options set. Visual Progress Window will launch shortly.\n');

%% SECTION 7: Training Execution
fprintf('\n[Step 7] Starting Training...\n');
fprintf('   > ------------------------------------------------------------\n');
fprintf('   > NOTE: A separate window will open to show the Progress Bar.\n');
fprintf('   > Look at the top-right of that window for "Estimated Time".\n');
fprintf('   > The Command Window will remain paused until training finishes.\n');
fprintf('   > ------------------------------------------------------------\n');

trainingTimer = tic; % Start timer

try
    trainedNet = trainNetwork(auimdsTrain, lgraph, options);
catch ME
    fprintf(2, '\nERROR DURING TRAINING: %s\n', ME.message);
    rethrow(ME);
end

trainingTime = toc(trainingTimer); % Stop timer
fprintf('\n   > Training Complete in %.2f minutes.\n', trainingTime/60);

%% SECTION 8: Save Result
fprintf('\n[Step 8] Saving Model...\n');
saveFilename = 'ASL_Trained_Network.mat';
save(saveFilename, 'trainedNet');

fprintf('   > Network saved to: %s\n', fullfile(pwd, saveFilename));
fprintf('=== SUCCESS: Script Finished ===\n');