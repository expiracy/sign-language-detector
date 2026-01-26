% Image_Processing_Explorer.m
% Simple script for testing a custom image processing pipeline
clc; clear; close all;

%% SECTION 1: Load Images
fprintf('[Step 1] Loading Images...\n');

imagePath = fullfile(pwd, 'data/');
imds = imageDatastore(imagePath, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.gif'});

numImages = numel(imds.Files);
fprintf(' > Found %d images.\n', numImages);

if numImages == 0
    error('No images found in the specified directory.');
end

%% SECTION 2: Select Samples
fprintf('\n[Step 2] Selecting Samples...\n');

numSamples = min(4, numImages);
sampleIndices = randperm(numImages, numSamples);

originalImages = cell(1, numSamples);
for i = 1:numSamples
    originalImages{i} = readAndNormalise(imds.Files{sampleIndices(i)});
end

fprintf(' > Selected %d sample images.\n', numSamples);

%% SECTION 3: Apply Pipeline and Visualise
fprintf('\n[Step 3] Applying Pipeline...\n');

figure('Name', 'Pipeline Results', 'Position', [100 100 1200 600]);

for i = 1:numSamples
    subplot(2, numSamples, i);
    imshow(originalImages{i});
    title('Original');
    
    subplot(2, numSamples, i + numSamples);
    processed = processingPipeline(originalImages{i});
    imshow(processed);
    title('Processed');
end

sgtitle('Custom Pipeline Results');
fprintf(' > Complete.\n');

%% ========== EDIT PIPELINE HERE ==========

function img = processingPipeline(img)
    img = gentleUnsharpMask(img);
    img = subtleLocalContrast(img);
end

%% ========== HELPER FUNCTIONS ==========

function img = readAndNormalise(filename)
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

function img = gentleUnsharpMask(img)
    blurred = imgaussfilt(img, 1.5);
    img = img + 0.3 * (img - blurred);
    img = max(0, min(1, img));
end

function img = subtleLocalContrast(img)
    lab = rgb2lab(img);
    L = lab(:,:,1) / 100;
    L = adapthisteq(L, 'ClipLimit', 0.005, 'NumTiles', [8 8]);
    lab(:,:,1) = L * 100;
    img = lab2rgb(lab);
    img = max(0, min(1, img));
end