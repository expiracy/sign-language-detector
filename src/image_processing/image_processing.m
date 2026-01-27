% Preprocess_ASL_Images.m
% Preprocesses images and saves them to folders based on the first letter of each filename.
clc; clear; close all;

%% Configuration
inputFolder = fullfile(pwd, 'data');
outputFolder = fullfile(pwd, 'data/preprocessed');
previewCount = 100;

%% Validate Input Folder
fprintf('[Step 1] Validating Input Folder...\n');
if ~exist(inputFolder, 'dir')
    error('Input folder not found: %s', inputFolder);
end

imageFiles = dir(fullfile(inputFolder, '**', '*.*'));
imageFiles = imageFiles(~[imageFiles.isdir]);
validExtensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

isImage = false(length(imageFiles), 1);
for i = 1:length(imageFiles)
    [~, ~, ext] = fileparts(imageFiles(i).name);
    isImage(i) = any(strcmpi(ext, validExtensions));
end
imageFiles = imageFiles(isImage);

fprintf('   > Found %d images in input folder.\n', length(imageFiles));
if isempty(imageFiles)
    error('No valid images found in input folder.');
end

%% Preview Preprocessing
fprintf('\n[Step 2] Generating Preview...\n');
numPreview = min(previewCount, length(imageFiles));
previewIndices = randperm(length(imageFiles), numPreview);

originalImages = cell(numPreview, 1);
processedImages = cell(numPreview, 1);

for i = 1:numPreview
    filePath = fullfile(imageFiles(previewIndices(i)).folder, imageFiles(previewIndices(i)).name);
    try
        img = imread(filePath);
        originalImages{i} = imresize(img, [128 128]);
        processedImages{i} = imresize(preprocessImage(img), [128 128]);
    catch
        originalImages{i} = uint8(zeros(128, 128, 3));
        processedImages{i} = uint8(zeros(128, 128, 3));
    end
end

fprintf('   > Displaying %d sample images. Close the figure to continue.\n', numPreview);

fig = figure('Name', 'Preprocessing Preview - Close to Continue', 'NumberTitle', 'off');
fig.Position = [100 100 1200 600];

subplot(1, 2, 1);
montage(originalImages, 'Size', [10 10], 'BorderSize', 2, 'BackgroundColor', 'white');
title('Original Images');

subplot(1, 2, 2);
montage(processedImages, 'Size', [10 10], 'BorderSize', 2, 'BackgroundColor', 'white');
title('Preprocessed Images');

sgtitle('Close this figure to proceed with full processing');

waitfor(fig);

%% Create Output Directory Structure
fprintf('\n[Step 3] Creating Output Directories...\n');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for letter = 'A':'Z'
    letterFolder = fullfile(outputFolder, letter);
    if ~exist(letterFolder, 'dir')
        mkdir(letterFolder);
    end
end
fprintf('   > Created A-Z subdirectories in: %s\n', outputFolder);

%% Process and Save Images
fprintf('\n[Step 4] Processing Images...\n');
totalImages = length(imageFiles);
processedCount = 0;
skippedCount = 0;

for i = 1:totalImages
    filePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    [~, baseName, ext] = fileparts(imageFiles(i).name);
    
    if isempty(baseName)
        skippedCount = skippedCount + 1;
        continue;
    end
    
    firstChar = upper(baseName(1));
    if firstChar < 'A' || firstChar > 'Z'
        fprintf('   > Skipping (invalid label): %s\n', imageFiles(i).name);
        skippedCount = skippedCount + 1;
        continue;
    end
    
    try
        img = imread(filePath);
        processedImg = preprocessImage(img);
        outputPath = fullfile(outputFolder, firstChar, [baseName, ext]);
        imwrite(processedImg, outputPath);
        processedCount = processedCount + 1;
        
        if mod(processedCount, 500) == 0
            fprintf('   > Processed %d / %d images...\n', processedCount, totalImages);
        end
    catch ME
        fprintf('   > Error processing %s: %s\n', imageFiles(i).name, ME.message);
        skippedCount = skippedCount + 1;
    end
end

%% Summary
fprintf('\n[Step 5] Complete.\n');
fprintf('   > Images processed: %d\n', processedCount);
fprintf('   > Images skipped:   %d\n', skippedCount);
fprintf('   > Output location:  %s\n', outputFolder);

%% Preprocessing Function
function img = preprocessImage(img)
    % Convert to LAB to handle luminance separately from colour
    lab = rgb2lab(img);
    L = lab(:,:,1) / 100;
    
    % Adaptive illumination correction via CLAHE
    L = adapthisteq(L, 'ClipLimit', 0.01, 'NumTiles', [4 4], 'Distribution', 'uniform');
    
    % Gentle homomorphic filtering for residual shadows
    logL = log1p(L);
    illumination = imgaussfilt(logL, 15);
    reflectance = logL - illumination;
    L = expm1(0.7 * illumination + 1.2 * reflectance);
    L = rescale(L);
    
    % Mild sharpening for edge definition
    L = imsharpen(L, 'Radius', 1, 'Amount', 0.6, 'Threshold', 0.04);
    
    lab(:,:,1) = L * 100;
    img = lab2rgb(lab);
    img = im2uint8(img);
end