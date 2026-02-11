classdef data_tester < matlab.apps.AppBase

    % App Components
    properties (Access = public)
        UIFigure                       matlab.ui.Figure
        ResultsPanel                   matlab.ui.container.Panel
        ShowConfusionMatrixButton      matlab.ui.control.Button
        F1ScoreLabel                   matlab.ui.control.Label
        RecallLabel                    matlab.ui.control.Label
        PrecisionLabel                 matlab.ui.control.Label
        AccuracyLabel                  matlab.ui.control.Label
        StatusLabel                    matlab.ui.control.Label
        CSVFileLabel                   matlab.ui.control.Label
        InputPanel                     matlab.ui.container.Panel
        UploadCSVButton                matlab.ui.control.Button
        ExpectedLetterDropDown         matlab.ui.control.DropDown
        ExpectedLetterDropDown_2Label  matlab.ui.control.Label
        FileLabel                      matlab.ui.control.Label
        RunEvaluationButton            matlab.ui.control.Button
        LoadNetworkButton              matlab.ui.control.Button
        SelectImageFolderButton        matlab.ui.control.Button
        UploadVideoButton              matlab.ui.control.Button
        TestInputFormatButtonGroup     matlab.ui.container.ButtonGroup
        ImageFolderButton              matlab.ui.control.RadioButton
        VideoCSVButton                 matlab.ui.control.RadioButton
        FolderPairsButton              matlab.ui.control.RadioButton
        FoundFilesListBox              matlab.ui.control.ListBox
        LoadedNetworkLabel             matlab.ui.control.Label
        NetworkNameEditField           matlab.ui.control.EditField
        NetworkNameLabel               matlab.ui.control.Label
        ModelNameResultLabel           matlab.ui.control.Label
    end

    
    properties (Access = private)

        % Data storage
        videoPath
        csvPath
        imageFolderPath
        modelPath

        % Found pairs for batch processing
        foundPairs = struct('csv', {}, 'mp4', {})
        
        % Added UI Component
        ClearFilesButton               matlab.ui.control.Button

        % Data
        frames          % Images
        trueLabels      % Labels
        predictions     % Predictions

        % Model
        Net             % Network
        NetInputSize    double = []
        NetClasses      % Classes

        % Results
        confusionMat
        accuracy
        precision
        recall
        f1score
    end
    
    methods (Access = private)
        
        function NetSelectionUpdated(app, fullPath)
            try
                data = load(fullPath);
                [net, ~] = app.findNetworkInLoadedStruct(data);
                if ~(isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork'))
                    error('Unsupported network type. Save a SeriesNetwork or DAGNetwork.');
                end
                app.Net = net;
                app.NetInputSize = app.getNetInputSize(net);
                
                % Update UI
                [~, fname, ~] = fileparts(fullPath);
                app.LoadedNetworkLabel.Text = fullPath;
                app.NetworkNameEditField.Value = fname;
                app.ModelNameResultLabel.Text = ['Model: ' fullPath];
                
                app.StatusLabel.Text = 'Network loaded.';
                app.StatusLabel.FontColor = [0 1 0];
                app.LoadNetworkButton.FontColor = [0 1 0];
            catch err
                uialert(app.UIFigure, err.message, 'Error');
            end
        end

        function [net, netVarName] = findNetworkInLoadedStruct(~, data)
            varNames = fieldnames(data);
            for i = 1:numel(varNames)
                v = data.(varNames{i});
                if isa(v, 'SeriesNetwork') || isa(v, 'DAGNetwork') || isa(v, 'dlnetwork')
                    net = v;
                    netVarName = string(varNames{i});
                    return;
                end
            end

            error("No neural network found in the selected .mat file.");
        end

        function inputSize = getNetInputSize(~, net)
            layers = net.Layers;
            for i = 1:numel(layers)
                if isprop(layers(i), 'InputSize')
                    sz = layers(i).InputSize;
                    if numel(sz) >= 2
                        inputSize = sz(1:2);
                        return;
                    end
                end
            end
            error("Could not find an InputSize in the network layers.");
        end


        

        function runPredictions(app)
            if isempty(app.Net)
                uialert(app.UIFigure, 'Load network first.', 'Error');
                return;
            end
            
            app.predictions = {};
            numFrames = length(app.frames);
            
            app.StatusLabel.Text = 'Predicting...';
            app.StatusLabel.FontColor = [0 0 1];
            drawnow;
            
            % Get all classes
            app.NetClasses = app.Net.Layers(end).Classes;
            numClasses = length(app.NetClasses);
            
            for i = 1:numFrames
                % Preprocess frame
                imgResized = imresize(app.frames{i}, app.NetInputSize);
                
                % Use classify()
                [labelCat, ~] = classify(app.Net, imgResized);
                
                % Store prediction
                prediction = char(labelCat);
                app.predictions{end+1} = prediction;
                
                % Update progress
                if mod(i, 10) == 0
                    app.StatusLabel.Text = sprintf('[%d/%d] Processing...', i, numFrames);
                    drawnow;
                end
            end
            
            app.StatusLabel.Text = 'Predictions done.';
            app.StatusLabel.FontColor = [0 1 0];
        end

        function loadVideoData(app, videoPathArg, csvPathArg, appendData)
            if nargin < 4
                 appendData = false;
            end
            if nargin < 3
                 csvPathArg = app.csvPath;
            end
            if nargin < 2
                 videoPathArg = app.videoPath;
            end

            try
                % Read CSV
                data = readtable(csvPathArg);
                
                % Validate CSV format
                requiredCols = {'Letter', 'StartFrame', 'EndFrame'};
                if ~all(ismember(requiredCols, data.Properties.VariableNames))
                    error('CSV must contain columns: Letter, StartFrame, EndFrame');
                end
                
                % Remove the TOTAL row if it exists
                totalIdx = strcmpi(data.Letter, 'TOTAL');
                if any(totalIdx)
                    data(totalIdx, :) = [];
                end
                
                % Open video
                if ~isfile(videoPathArg)
                    error('Video file not found: %s', videoPathArg);
                end
                vid = VideoReader(videoPathArg);
                
                if ~appendData
                    app.frames = {};
                    app.trueLabels = {};
                end
                
                app.StatusLabel.Text = 'Loading frames...';
                drawnow;
                
                % Extract frames for each letter
                for i = 1:height(data)
                    letter = data.Letter{i};
                    startFrame = data.StartFrame(i);
                    endFrame = data.EndFrame(i);
                    
                    % Validate frame numbers
                    if startFrame < 1 || endFrame > vid.NumFrames
                        warning('Skipping %s: frames %d-%d out of range (video has %d frames)', ...
                            letter, startFrame, endFrame, vid.NumFrames);
                        continue;
                    end
                    
                    % Read frames from startFrame to endFrame
                    for frameNum = startFrame:endFrame
                        vid.CurrentTime = (frameNum - 1) / vid.FrameRate;
                        if hasFrame(vid)
                            frame = readFrame(vid);
                            % Store frame and true label
                            app.frames{end+1} = frame;
                            app.trueLabels{end+1} = letter;
                        end
                    end
                end
                
                if isempty(app.frames)
                    error('No frames loaded.');
                end
                
                app.StatusLabel.Text = sprintf('Loaded %d frames', length(app.frames));
                app.StatusLabel.FontColor = [0 1 0];
                
            catch ME
                uialert(app.UIFigure, ME.message, 'Video Error');
                app.StatusLabel.Text = 'Error loading.';
                app.StatusLabel.FontColor = [1 0 0];
            end
        end
        
        function loadImageData(app)
            % Get all image files
            imageFiles = dir(fullfile(app.imageFolderPath, '*.jpg'));
            if isempty(imageFiles)
                imageFiles = dir(fullfile(app.imageFolderPath, '*.png'));
            end
            
            if isempty(imageFiles)
                uialert(app.UIFigure, 'No images found in folder!', 'Error');
                return;
            end
            
            app.frames = {};
            app.trueLabels = {};
            expectedLetter = app.ExpectedLetterDropDown.Value;
            
            app.StatusLabel.Text = 'Loading images...';
            drawnow;
            
            for i = 1:length(imageFiles)
                img = imread(fullfile(app.imageFolderPath, imageFiles(i).name));
                app.frames{end+1} = img;
                app.trueLabels{end+1} = expectedLetter;
            end
            
            app.StatusLabel.Text = sprintf('Loaded %d images', length(app.frames));
        end
        
        function calculateMetrics(app)
            % Check if we have predictions
            if isempty(app.predictions) || isempty(app.trueLabels)
                error('No predictions or true labels available');
            end
            
            % Ensure same length
            minLen = min(length(app.predictions), length(app.trueLabels));
            predictions = app.predictions(1:minLen);
            trueLabels = app.trueLabels(1:minLen);
            
            % Get ALL 24 classes from network
            allClassNames = cellstr(app.NetClasses);
            
            % Convert to categorical with ALL 24 classes
            true_cat = categorical(trueLabels, allClassNames);
            pred_cat = categorical(predictions, allClassNames);
            
            % Create 24x24 confusion matrix
            app.confusionMat = confusionmat(true_cat, pred_cat);
            
            % Overall accuracy
            app.accuracy = sum(strcmp(trueLabels, predictions)) / minLen;
            
            % Calculate per-class metrics (macro-averaged)
            testedClasses = unique(trueLabels);
            numTestedClasses = length(testedClasses);
            
            precisions = zeros(numTestedClasses, 1);
            recalls = zeros(numTestedClasses, 1);
            
            for i = 1:numTestedClasses
                label = testedClasses{i};
                
                TP = sum(strcmp(trueLabels, label) & strcmp(predictions, label));
                FP = sum(~strcmp(trueLabels, label) & strcmp(predictions, label));
                FN = sum(strcmp(trueLabels, label) & ~strcmp(predictions, label));
                        
                if (TP + FP) > 0
                    precisions(i) = TP / (TP + FP);
                else
                    precisions(i) = 0;
                end
                
                if (TP + FN) > 0
                    recalls(i) = TP / (TP + FN);
                else
                    recalls(i) = 0;
                end
            end
            
            % Macro-averaged metrics (ignore NaN values)
            app.precision = mean(precisions);
            app.recall = mean(recalls);
            
            if (app.precision + app.recall) > 0
                app.f1score = 2 * (app.precision * app.recall) / (app.precision + app.recall);
            else
                app.f1score = 0;
            end
            
            % Display which classes were actually tested
            app.StatusLabel.Text = sprintf('Tested %d classes: %s', ...
                numTestedClasses, strjoin(testedClasses, ', '));
        end
        
        function displayResults(app)
            % Update Model Name result
            name = app.NetworkNameEditField.Value;
            if isempty(name), name = '-'; end
            app.ModelNameResultLabel.Text = ['Model: ' name];

            app.AccuracyLabel.Text = sprintf('Accuracy: %.2f%%', app.accuracy * 100);
            app.PrecisionLabel.Text = sprintf('Precision: %.2f%%', app.precision * 100);
            app.RecallLabel.Text = sprintf('Recall: %.2f%%', app.recall * 100);
            app.F1ScoreLabel.Text = sprintf('F1-Score: %.2f%%', app.f1score * 100);
            app.StatusLabel.Text = 'Done.';
            app.StatusLabel.FontColor = [0 1 0];
        end

    end
    

    % Callbacks
    methods (Access = private)

        % Startup
        function startupFcn(app)
            % Initialize with Video mode selected
            app.VideoCSVButton.Value = true;
            
            % Trigger the selection change callback to set initial visibility correctly
            TestInputFormatButtonGroupSelectionChanged(app, []);
            
            % Initialize status
            app.StatusLabel.Text = 'Ready.';
            app.StatusLabel.FontColor = [0 0 0];
        end

        % Load Network
        function LoadNetworkButtonPushed(app, event)
            [file, path] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select a trained network');
            if isequal(file, 0)
                return;
            end
            fullPath = fullfile(path, file);
            app.modelPath = fullPath;
            NetSelectionUpdated(app, fullPath);
        end

        % Upload CSV
        function UploadCSVButtonPushed(app, event)
            [file, path] = uigetfile('*.csv', 'Select CSV File');
            if file ~= 0
                app.csvPath = fullfile(path, file);
                app.StatusLabel.Text = ['CSV: ' file];
                app.StatusLabel.FontColor = [0 0 1];

                app.CSVFileLabel.Text = file;
                app.UploadCSVButton.FontColor = [0 1 0];
            end
        end

        % Select Folder
        function SelectImageFolderButtonPushed(app, event)
            folder = uigetdir(pwd, 'Select Folder');
            if folder ~= 0
                app.imageFolderPath = folder;

                if app.FolderPairsButton.Value
                    app.FileLabel.Text = folder; % Show folder path
                    app.SelectImageFolderButton.FontColor = [0 1 0];
                    
                    % Scan for pairs
                    app.StatusLabel.Text = 'Scanning...';
                    drawnow;
                    
                    % Recursive search
                    csvFiles = dir(fullfile(app.imageFolderPath, '**', '*.csv'));
                    
                    % Append pairs
                    listBoxItems = app.FoundFilesListBox.Items;
                    newPairsCount = 0;
                    
                    for i = 1:length(csvFiles)
                        csvName = csvFiles(i).name;
                        csvFolder = csvFiles(i).folder;
                        
                        % Check pattern *_<number>.csv
                        tokens = regexp(csvName, '(.*)_(\d+)\.csv$', 'tokens');
                        if isempty(tokens)
                            continue;
                        end
                        
                        numberPart = tokens{1}{2};
                        
                        % Look for matching MP4
                        mp4Pattern = sprintf('*_%s.mp4', numberPart);
                        mp4Files = dir(fullfile(csvFolder, mp4Pattern));
                        
                        if ~isempty(mp4Files)
                            mp4Name = mp4Files(1).name;
                            
                            % Store pair
                            app.foundPairs(end+1).csv = fullfile(csvFolder, csvName);
                            app.foundPairs(end).mp4 = fullfile(csvFolder, mp4Name);
                            
                            % Display with parent folder
                            [~, immediateParent, ~] = fileparts(csvFolder);
                            listBoxItems{end+1} = sprintf('[%s] %s', immediateParent, csvName);
                            newPairsCount = newPairsCount + 1;
                        end
                    end
                    
                    app.FoundFilesListBox.Items = listBoxItems;
                    
                    if isempty(app.foundPairs)
                        if isempty(listBoxItems)
                             app.StatusLabel.Text = 'No pairs found.';
                             app.StatusLabel.FontColor = [1 0 0];
                        else
                             app.StatusLabel.Text = sprintf('Found %d pairs.', length(app.foundPairs));
                             app.StatusLabel.FontColor = [1 0.5 0];
                        end
                    else
                        app.StatusLabel.Text = sprintf('Added %d. Total: %d', newPairsCount, length(app.foundPairs));
                        app.StatusLabel.FontColor = [0 1 0];
                    end
                    
                else
                    app.FileLabel.Text = folder;
                    app.SelectImageFolderButton.FontColor = [0 1 0];
                    app.StatusLabel.Text = 'Folder selected.';
                end
            end
        end

        % Run Evaluation
        function RunEvaluationButtonPushed(app, event)
            if isempty(app.Net)
                uialert(app.UIFigure, 'Load network first.', 'Error');
                return;
            end
            
            % Reset results
            app.AccuracyLabel.Text = 'Accuracy';
            app.PrecisionLabel.Text = 'Precision';
            app.RecallLabel.Text = 'Recall';
            app.F1ScoreLabel.Text = 'F1-Score';
            app.confusionMat = [];
            
            try
                % Check mode
                if app.VideoCSVButton.Value
                    if isempty(app.videoPath) || isempty(app.csvPath)
                        uialert(app.UIFigure, 'Select video and CSV.', 'Error');
                        return;
                    end
                    loadVideoData(app);
                    
                elseif app.FolderPairsButton.Value
                    if isempty(app.foundPairs)
                        uialert(app.UIFigure, 'No pairs found.', 'Error');
                        return;
                    end
                    
                    app.StatusLabel.Text = 'Batching...';
                    drawnow;
                    
                    % Accumulate stats
                    allPredictions = {};
                    allTrueLabels = {};
                    filesProcessed = 0;
                    totalFiles = length(app.foundPairs);
                    
                    for i = 1:totalFiles
                        try
                            pair = app.foundPairs(i);
                            [~, name, ~] = fileparts(pair.csv);
                            
                            app.StatusLabel.Text = sprintf('[%d/%d] %s', i, totalFiles, name);
                            drawnow;
                            
                            % Load and predict
                            loadVideoData(app, pair.mp4, pair.csv, false);
                            runPredictions(app);
                            
                            % Store
                            allPredictions = [allPredictions, app.predictions]; %#ok<AGROW>
                            allTrueLabels = [allTrueLabels, app.trueLabels]; %#ok<AGROW>
                            
                            filesProcessed = filesProcessed + 1;
                            
                        catch ME
                            warning('Skipped pair %d: %s', i, ME.message);
                        end
                    end
                    
                    if filesProcessed == 0
                         error('No pairs processed.');
                    end
                    
                    % Aggregate
                    app.predictions = allPredictions;
                    app.trueLabels = allTrueLabels;
                    
                    app.StatusLabel.Text = sprintf('Processed %d. Calculating...', filesProcessed);
                    
                else
                    if isempty(app.imageFolderPath)
                        uialert(app.UIFigure, 'Select an image folder.', 'Error');
                        return;
                    end
                    loadImageData(app);
                    runPredictions(app);
                end
                
                % Video mode needs prediction call
                if app.VideoCSVButton.Value
                     runPredictions(app);
                end

                % Calculate metrics
                calculateMetrics(app);
                
                % Display results
                displayResults(app);
                
            catch ME
                uialert(app.UIFigure, ME.message, 'Error');
                app.StatusLabel.Text = 'Error.';
                app.StatusLabel.FontColor = [1 0 0];
            end
        end

        % Confusion Matrix
        function ShowConfusionMatrixButtonPushed(app, event)
            if isempty(app.Net)
                uialert(app.UIFigure, 'Load network first.', 'Error');
                return;
            end
            
            % Check results
            if isempty(app.predictions) || isempty(app.trueLabels)
                uialert(app.UIFigure, 'Run evaluation first.', 'Info');
                return;
            end
            
            % Get all classes
            allClassNames = cellstr(app.NetClasses);
            
            % Ensure same length
            minLen = min(length(app.predictions), length(app.trueLabels));
            predictions = app.predictions(1:minLen);
            trueLabels = app.trueLabels(1:minLen);
            
            % Convert to categorical
            true_cat = categorical(trueLabels, allClassNames);
            pred_cat = categorical(predictions, allClassNames);
            
            % Create figure
            fig = figure('Name', 'Confusion Matrix', ...
                        'NumberTitle', 'off', ...
                        'Position', [100, 100, 1200, 900]);
            
            % Chart
            cm = confusionchart(true_cat, pred_cat);
            
            % Customize
            netName = app.NetworkNameEditField.Value;
            if isempty(netName)
                netName = 'ASL Model';
            end
            
            cm.Title = sprintf('%s - Confusion Matrix (Acc: %.1f%%)', ...
                netName, app.accuracy * 100);
            cm.RowSummary = 'row-normalized';
            cm.ColumnSummary = 'column-normalized';
            cm.FontSize = 10;
                       
            colormap(jet);
            
            % Annotations
            annotation('textbox', [0.1, 0.02, 0.8, 0.05], ...
                'String', 'Rows: True Labels | Columns: Predicted Labels | Diagonal: Correct Predictions', ...
                'EdgeColor', 'none', ...
                'FontSize', 10, ...
                'HorizontalAlignment', 'center');
            
            % Save to tests folder
            try
                % Determine project root (../../ from this file)
                [thisPath, ~, ~] = fileparts(mfilename('fullpath'));
                projectRoot = fileparts(fileparts(thisPath));
                testsFolder = fullfile(projectRoot, 'outputs', 'tests');
                
                if ~exist(testsFolder, 'dir')
                    mkdir(testsFolder);
                end
                
                % Clean filename
                safeName = regexprep(netName, '[\\/:*?"<>|]', '_');
                savePath = fullfile(testsFolder, [safeName '.png']);
                
                saveas(fig, savePath);
                fprintf('Saved confusion matrix: %s\n', savePath);
                app.StatusLabel.Text = ['Saved: ' safeName '.png'];
            catch outputErr
                warning(outputErr.message);
            end
            
            % Debug print
            fprintf('\nMatrix\n');
            fprintf('Size: %dx%d\n', size(app.confusionMat));
            fprintf('Classes: %s\n', strjoin(allClassNames, ', '));
            fprintf('\n');
            
            % Display confusion matrix in command window
            disp('Confusion Matrix (True x Predicted):');
            fprintf('     ');
            for i = 1:min(10, length(allClassNames))
                fprintf('%4s ', allClassNames{i});
            end
            fprintf('\n');
            
            for i = 1:min(10, length(allClassNames))
                fprintf('%4s ', allClassNames{i});
                for j = 1:min(10, length(allClassNames))
                    fprintf('%4d ', app.confusionMat(i,j));
                end
                fprintf('\n');
            end
            
            if length(allClassNames) > 10
                fprintf('... (showing first 10x10)\n');
            end
        end

        % Input Format Selection Changed
        function TestInputFormatButtonGroupSelectionChanged(app, event)
            if isempty(event) || ~isprop(event, 'NewValue')
                selectedButton = app.TestInputFormatButtonGroup.SelectedObject;
            else
                selectedButton = event.NewValue;
            end
            
            % Mode check
            isFolderPairs = strcmp(selectedButton.Text, 'Folder of Pairs');
            isImageFolder = strcmp(selectedButton.Text, 'Image Folder');
            isVideoMode = strcmp(selectedButton.Text, 'Video + CSV');
            
            % Visibility helper
            function s = bool2vis(b)
                if b, s = 'on'; else, s = 'off'; end
            end
            
            % 1. Video Controls
            app.UploadVideoButton.Visible = bool2vis(isVideoMode);
            app.UploadCSVButton.Visible = bool2vis(isVideoMode);
            app.CSVFileLabel.Visible = bool2vis(isVideoMode);
            
            % 2. Folder Button
            app.SelectImageFolderButton.Visible = bool2vis(isFolderPairs || isImageFolder);
            if isFolderPairs
                app.SelectImageFolderButton.Text = 'Add Data Folder';
            elseif isImageFolder
                app.SelectImageFolderButton.Text = 'Select Image Folder';
            end
            
            % 3. Mode Controls
            app.FoundFilesListBox.Visible = bool2vis(isFolderPairs);
            app.ClearFilesButton.Visible = bool2vis(isFolderPairs);
            app.ExpectedLetterDropDown.Visible = bool2vis(isImageFolder);
            app.ExpectedLetterDropDown_2Label.Visible = bool2vis(isImageFolder);
        end
        
        % Clear Files
        function ClearFilesButtonPushed(app, event)
            app.foundPairs = struct('csv', {}, 'mp4', {});
            app.FoundFilesListBox.Items = {};
            app.StatusLabel.Text = 'Cleared.';
            app.StatusLabel.FontColor = [0 0 0];
        end

        % Upload Video
        function UploadVideoButtonPushed(app, event)
            [file, path] = uigetfile({'*.mp4', ...
                'Video Files (*.mp4)'}, ...
                'Select Video File');
            if file ~= 0
                app.videoPath = fullfile(path, file);
                app.StatusLabel.Text = ['Video: ' file];
                app.StatusLabel.FontColor = [0 0 1];

                app.FileLabel.Text = file;
                app.UploadVideoButton.FontColor = [0 1 0];
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Initialize UI
        function createComponents(app)

            % Create Figure
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 540];
            app.UIFigure.Name = 'ASL Data Tester';

            % Main Grid
            mainGrid = uigridlayout(app.UIFigure);
            mainGrid.ColumnWidth = {'1.5x', '1x'}; 
            mainGrid.RowHeight = {'1x', 30};

            % -- Input Panel --
            app.InputPanel = uipanel(mainGrid);
            app.InputPanel.Title = 'Input';
            app.InputPanel.Layout.Row = 1;
            app.InputPanel.Layout.Column = 1;

            % Input Grid
            inputGrid = uigridlayout(app.InputPanel);
            inputGrid.ColumnWidth = {'1x', '1x'};
            % Rows: 1:Mode, 2:Video/Folder, 3:CSV/Letter, 4:Listbox, 5:LoadNet, 6:LoadedLbl, 7:NameEditor, 8:Eval
            inputGrid.RowHeight = {80, 40, 40, '1x', 40, 25, 30, 50}; 

            % 1. Input Format Group
            app.TestInputFormatButtonGroup = uibuttongroup(inputGrid);
            app.TestInputFormatButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @TestInputFormatButtonGroupSelectionChanged, true);
            app.TestInputFormatButtonGroup.Title = 'Test Input Format';
            app.TestInputFormatButtonGroup.Layout.Row = 1;
            app.TestInputFormatButtonGroup.Layout.Column = [1 2];

            % Radio Buttons (Manual positioning relative to Group)
            app.VideoCSVButton = uiradiobutton(app.TestInputFormatButtonGroup);
            app.VideoCSVButton.Text = 'Video + CSV';
            app.VideoCSVButton.Position = [10 35 100 22];
            app.VideoCSVButton.Value = true;

            app.FolderPairsButton = uiradiobutton(app.TestInputFormatButtonGroup);
            app.FolderPairsButton.Text = 'Folder of Pairs';
            app.FolderPairsButton.Position = [120 35 110 22];

            app.ImageFolderButton = uiradiobutton(app.TestInputFormatButtonGroup);
            app.ImageFolderButton.Text = 'Image Folder';
            app.ImageFolderButton.Position = [10 10 100 22];

            % 2. File Selection Row
            app.UploadVideoButton = uibutton(inputGrid, 'push');
            app.UploadVideoButton.Text = 'Upload Video';
            app.UploadVideoButton.Layout.Row = 2;
            app.UploadVideoButton.Layout.Column = 1;
            app.UploadVideoButton.ButtonPushedFcn = createCallbackFcn(app, @UploadVideoButtonPushed, true);

            app.SelectImageFolderButton = uibutton(inputGrid, 'push');
            app.SelectImageFolderButton.Text = 'Select Folder';
            app.SelectImageFolderButton.Layout.Row = 2;
            app.SelectImageFolderButton.Layout.Column = 1;
            app.SelectImageFolderButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImageFolderButtonPushed, true);
            app.SelectImageFolderButton.Visible = 'off';

            app.FileLabel = uilabel(inputGrid);
            app.FileLabel.Text = '';
            app.FileLabel.Layout.Row = 2;
            app.FileLabel.Layout.Column = 2;

            % Clear Files Button
            app.ClearFilesButton = uibutton(inputGrid, 'push');
            app.ClearFilesButton.Text = 'Clear List';
            app.ClearFilesButton.Layout.Row = 2;
            app.ClearFilesButton.Layout.Column = 2;
            app.ClearFilesButton.Visible = 'off';
            app.ClearFilesButton.ButtonPushedFcn = createCallbackFcn(app, @ClearFilesButtonPushed, true);

            % 3. Secondary Input Row (CSV or Expected Letter)
            app.UploadCSVButton = uibutton(inputGrid, 'push');
            app.UploadCSVButton.Text = 'Upload CSV';
            app.UploadCSVButton.Layout.Row = 3;
            app.UploadCSVButton.Layout.Column = 1;
            app.UploadCSVButton.ButtonPushedFcn = createCallbackFcn(app, @UploadCSVButtonPushed, true);

            app.CSVFileLabel = uilabel(inputGrid);
            app.CSVFileLabel.Text = '';
            app.CSVFileLabel.Layout.Row = 3;
            app.CSVFileLabel.Layout.Column = 2;

            app.ExpectedLetterDropDown_2Label = uilabel(inputGrid);
            app.ExpectedLetterDropDown_2Label.Text = 'Expected Letter:';
            app.ExpectedLetterDropDown_2Label.HorizontalAlignment = 'right';
            app.ExpectedLetterDropDown_2Label.Layout.Row = 3;
            app.ExpectedLetterDropDown_2Label.Layout.Column = 1;
            app.ExpectedLetterDropDown_2Label.Visible = 'off';

            app.ExpectedLetterDropDown = uidropdown(inputGrid);
            app.ExpectedLetterDropDown.Items = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'};
            app.ExpectedLetterDropDown.Value = 'A';
            app.ExpectedLetterDropDown.Layout.Row = 3;
            app.ExpectedLetterDropDown.Layout.Column = 2;
            app.ExpectedLetterDropDown.Visible = 'off';

            % 4. Found Pairs List Box
            app.FoundFilesListBox = uilistbox(inputGrid);
            app.FoundFilesListBox.Items = {};
            app.FoundFilesListBox.Layout.Row = 4;
            app.FoundFilesListBox.Layout.Column = [1 2];
            app.FoundFilesListBox.Visible = 'off';

            % 5. Load Network
            app.LoadNetworkButton = uibutton(inputGrid, 'push');
            app.LoadNetworkButton.Text = 'Load Network';
            app.LoadNetworkButton.Layout.Row = 5;
            app.LoadNetworkButton.Layout.Column = [1 2];
            app.LoadNetworkButton.ButtonPushedFcn = createCallbackFcn(app, @LoadNetworkButtonPushed, true);

            % 6. Loaded Network Label
            app.LoadedNetworkLabel = uilabel(inputGrid);
            app.LoadedNetworkLabel.Text = '(No network loaded)';
            app.LoadedNetworkLabel.HorizontalAlignment = 'center';
            app.LoadedNetworkLabel.FontColor = [0.5 0.5 0.5];
            app.LoadedNetworkLabel.Layout.Row = 6;
            app.LoadedNetworkLabel.Layout.Column = [1 2];

            % 7. Network Name Input
            app.NetworkNameLabel = uilabel(inputGrid);
            app.NetworkNameLabel.Text = 'Name:';
            app.NetworkNameLabel.HorizontalAlignment = 'right';
            app.NetworkNameLabel.Layout.Row = 7;
            app.NetworkNameLabel.Layout.Column = 1;

            app.NetworkNameEditField = uieditfield(inputGrid, 'text');
            app.NetworkNameEditField.Layout.Row = 7;
            app.NetworkNameEditField.Layout.Column = 2;
            app.NetworkNameEditField.Placeholder = 'Model Name';

            % 8. Run Evaluation
            app.RunEvaluationButton = uibutton(inputGrid, 'push');
            app.RunEvaluationButton.Text = 'Run Evaluation';
            app.RunEvaluationButton.FontSize = 18;
            app.RunEvaluationButton.Layout.Row = 8;
            app.RunEvaluationButton.Layout.Column = [1 2];
            app.RunEvaluationButton.ButtonPushedFcn = createCallbackFcn(app, @RunEvaluationButtonPushed, true);

            % -- Results Panel --
            app.ResultsPanel = uipanel(mainGrid);
            app.ResultsPanel.Title = 'Results';
            app.ResultsPanel.Layout.Row = 1;
            app.ResultsPanel.Layout.Column = 2;

            resultsGrid = uigridlayout(app.ResultsPanel);
            resultsGrid.ColumnWidth = {'1x'};
            resultsGrid.RowHeight = {30, 30, 30, 30, 30, 40, '1x'};

            app.ModelNameResultLabel = uilabel(resultsGrid);
            app.ModelNameResultLabel.Text = 'Model: -';
            app.ModelNameResultLabel.FontWeight = 'bold';
            app.ModelNameResultLabel.Layout.Row = 1;

            app.AccuracyLabel = uilabel(resultsGrid);
            app.AccuracyLabel.Text = 'Accuracy:';
            app.AccuracyLabel.Layout.Row = 2;

            app.PrecisionLabel = uilabel(resultsGrid);
            app.PrecisionLabel.Text = 'Precision:';
            app.PrecisionLabel.Layout.Row = 3;

            app.RecallLabel = uilabel(resultsGrid);
            app.RecallLabel.Text = 'Recall:';
            app.RecallLabel.Layout.Row = 4;

            app.F1ScoreLabel = uilabel(resultsGrid);
            app.F1ScoreLabel.Text = 'F1-Score:';
            app.F1ScoreLabel.Layout.Row = 5;

            app.ShowConfusionMatrixButton = uibutton(resultsGrid, 'push');
            app.ShowConfusionMatrixButton.Text = 'Show Confusion Matrix';
            app.ShowConfusionMatrixButton.Layout.Row = 6;
            app.ShowConfusionMatrixButton.ButtonPushedFcn = createCallbackFcn(app, @ShowConfusionMatrixButtonPushed, true);

            % -- Status Label --
            app.StatusLabel = uilabel(mainGrid);
            app.StatusLabel.Layout.Row = 2;
            app.StatusLabel.Layout.Column = [1 2];
            app.StatusLabel.Text = 'Note';
            
            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App Lifecycle
    methods (Access = public)

        % Constructor
        function app = data_tester

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Destructor
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end