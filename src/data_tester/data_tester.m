classdef data_tester < matlab.apps.AppBase

    % Properties that correspond to app components
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
    end

    
    properties (Access = private)

        % Data storage
        videoPath
        csvPath
        imageFolderPath
        modelPath

        % Found pairs for batch processing
        foundPairs = struct('csv', {}, 'mp4', {})

        % Processed data
        frames          % Cell array of images
        trueLabels      % Cell array of true labels
        predictions     % Cell array of predictions

        % Model
        Net % The trained AI network (SeriesNetwork or DAGNetwork)
        NetInputSize     double = [] % Expected input size [H W C]
        NetClasses       % All classes the network knows (24 letters)

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
                app.StatusLabel.Text = 'Network loaded successfully!';
                app.StatusLabel.FontColor = [0 1 0];
                app.LoadNetworkButton.FontColor = [0 1 0];
            catch err
                uialert(app.UIFigure, err.message, 'Network Load Error');
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
                uialert(app.UIFigure, 'Please load a network first!', 'Error');
                return;
            end
            
            app.predictions = {};
            numFrames = length(app.frames);
            
            app.StatusLabel.Text = 'Running predictions...';
            app.StatusLabel.FontColor = [0 0 1];
            drawnow;
            
            % Get all 24 classes from network
            app.NetClasses = app.Net.Layers(end).Classes;
            numClasses = length(app.NetClasses);
            
            for i = 1:numFrames
                % Preprocess frame to match network input
                imgResized = imresize(app.frames{i}, app.NetInputSize);
                
                % Use classify() - this predicts against ALL 24 classes
                [labelCat, ~] = classify(app.Net, imgResized);
                
                % Store prediction (this will be one of the 24 letters)
                prediction = char(labelCat);
                app.predictions{end+1} = prediction;
                
                % Update progress
                if mod(i, 10) == 0
                    app.StatusLabel.Text = sprintf('Processing... %d/%d frames', i, numFrames);
                    drawnow;
                end
            end
            
            app.StatusLabel.Text = 'Predictions complete!';
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
                
                app.StatusLabel.Text = 'Loading video frames...';
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
                    error('No frames were loaded from the video!');
                end
                
                app.StatusLabel.Text = sprintf('Loaded %d frames total', length(app.frames));
                app.StatusLabel.FontColor = [0 1 0];
                
            catch ME
                uialert(app.UIFigure, ME.message, 'Video Loading Error');
                app.StatusLabel.Text = 'Error loading video data';
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
            
            % Create 24×24 confusion matrix
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
            app.StatusLabel.Text = sprintf('Tested %d of %d classes: %s', ...
                numTestedClasses, length(allClassNames), strjoin(testedClasses, ', '));
        end
        
        function displayResults(app)
            app.AccuracyLabel.Text = sprintf('Accuracy: %.2f%%', app.accuracy * 100);
            app.PrecisionLabel.Text = sprintf('Precision: %.2f%%', app.precision * 100);
            app.RecallLabel.Text = sprintf('Recall: %.2f%%', app.recall * 100);
            app.F1ScoreLabel.Text = sprintf('F1-Score: %.2f%%', app.f1score * 100);
            app.StatusLabel.Text = 'Evaluation complete!';
            app.StatusLabel.FontColor = [0 1 0];
        end

    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Initialize with Video mode selected
            app.VideoCSVButton.Value = true;
            
            % Trigger the selection change callback to set initial visibility correctly
            TestInputFormatButtonGroupSelectionChanged(app, []);
            
            % Initialize status
            app.StatusLabel.Text = 'Ready - Select input type and load network';
            app.StatusLabel.FontColor = [0 1 0];
        end

        % Button pushed function: LoadNetworkButton
        function LoadNetworkButtonPushed(app, event)
            [file, path] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select a trained network');
            if isequal(file, 0)
                return;
            end
            fullPath = fullfile(path, file);
            app.modelPath = fullPath;
            NetSelectionUpdated(app, fullPath);
        end

        % Button pushed function: UploadCSVButton
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

        % Button pushed function: SelectImageFolderButton
        function SelectImageFolderButtonPushed(app, event)
            folder = uigetdir(pwd, 'Select Folder');
            if folder ~= 0
                app.imageFolderPath = folder;

                if app.FolderPairsButton.Value
                    app.FileLabel.Text = folder; % Show folder path
                    app.SelectImageFolderButton.FontColor = [0 1 0];
                    
                    % Scan for pairs
                    app.StatusLabel.Text = 'Scanning for file pairs...';
                    drawnow;
                    
                    % Find CSV files matching pattern
                    csvFiles = dir(fullfile(app.imageFolderPath, '*.csv'));
                    app.foundPairs = struct('csv', {}, 'mp4', {});
                    listBoxItems = {};
                    
                    for i = 1:length(csvFiles)
                        csvName = csvFiles(i).name;
                        % Check if it matches pattern *_<number>.csv
                        tokens = regexp(csvName, '(.*)_(\d+)\.csv$', 'tokens');
                        if isempty(tokens)
                            continue;
                        end
                        
                        numberPart = tokens{1}{2};
                        
                        % Look for matching MP4: *_<number>.mp4
                        mp4Files = dir(fullfile(app.imageFolderPath, sprintf('*_%s.mp4', numberPart)));
                        
                        if ~isempty(mp4Files)
                            mp4Name = mp4Files(1).name;
                            
                            % Store pair
                            app.foundPairs(end+1).csv = fullfile(app.imageFolderPath, csvName);
                            app.foundPairs(end).mp4 = fullfile(app.imageFolderPath, mp4Name);
                            
                            % Add to display list
                            listBoxItems{end+1} = sprintf('%s  <-->  %s', csvName, mp4Name);
                        end
                    end
                    
                    app.FoundFilesListBox.Items = listBoxItems;
                    
                    if isempty(app.foundPairs)
                        app.StatusLabel.Text = 'No matching pairs found in folder.';
                        app.StatusLabel.FontColor = [1 0 0];
                    else
                        app.StatusLabel.Text = sprintf('Found %d pairs.', length(app.foundPairs));
                        app.StatusLabel.FontColor = [0 1 0];
                    end
                    
                else
                    % Standard image folder mode
                    app.FileLabel.Text = folder;
                    app.SelectImageFolderButton.FontColor = [0 1 0];
                    app.StatusLabel.Text = 'Image folder selected.';
                end
            end
        end

        % Button pushed function: RunEvaluationButton
        function RunEvaluationButtonPushed(app, event)
            if isempty(app.Net)
                uialert(app.UIFigure, 'Please load a network first!', 'Error');
                return;
            end
            
            % Clear previous results
            app.AccuracyLabel.Text = 'Accuracy';
            app.PrecisionLabel.Text = 'Precision';
            app.RecallLabel.Text = 'Recall';
            app.F1ScoreLabel.Text = 'F1-Score';
            app.confusionMat = [];
            
            try
                % 1. Load data based on input type
                if app.VideoCSVButton.Value
                    if isempty(app.videoPath) || isempty(app.csvPath)
                        uialert(app.UIFigure, 'Please select both video and CSV files!', 'Error');
                        return;
                    end
                    loadVideoData(app);
                    
                elseif app.FolderPairsButton.Value
                    if isempty(app.foundPairs)
                        uialert(app.UIFigure, 'No data pairs found! Select a valid folder first.', 'Error');
                        return;
                    end
                    
                    app.StatusLabel.Text = 'Processing pairs...';
                    drawnow;
                    
                    app.frames = {};
                    app.trueLabels = {};
                    
                    filesLoaded = 0;
                    
                    for i = 1:length(app.foundPairs)
                        try
                            loadVideoData(app, app.foundPairs(i).mp4, app.foundPairs(i).csv, true);
                            filesLoaded = filesLoaded + 1;
                        catch ME
                            warning('Failed to load pair %d: %s', i, ME.message);
                        end
                    end
                    
                    if filesLoaded == 0
                         error('Failed to load any of the found pairs.');
                    end
                    
                    app.StatusLabel.Text = sprintf('Loaded data from %d file pairs', filesLoaded);
                    
                else
                    if isempty(app.imageFolderPath)
                        uialert(app.UIFigure, 'Please select an image folder!', 'Error');
                        return;
                    end
                    loadImageData(app);
                end
                
                % 2. Run predictions (against ALL 24 classes)
                runPredictions(app);
                
                % 3. Calculate metrics
                calculateMetrics(app);
                
                % 4. Display results
                displayResults(app);
                
            catch ME
                uialert(app.UIFigure, ME.message, 'Evaluation Error');
                app.StatusLabel.Text = 'Error during evaluation';
                app.StatusLabel.FontColor = [1 0 0];
            end
        end

        % Button pushed function: ShowConfusionMatrixButton
        function ShowConfusionMatrixButtonPushed(app, event)
            if isempty(app.Net)
                uialert(app.UIFigure, 'Load a network first!', 'Error');
                return;
            end
            
            % Check if we have predictions
            if isempty(app.predictions) || isempty(app.trueLabels)
                uialert(app.UIFigure, 'Run evaluation first to see confusion matrix!', 'Info');
                return;
            end
            
            % Get ALL 24 classes from the network
            allClassNames = cellstr(app.NetClasses);
            
            % Ensure same length
            minLen = min(length(app.predictions), length(app.trueLabels));
            predictions = app.predictions(1:minLen);
            trueLabels = app.trueLabels(1:minLen);
            
            % Convert to categorical with ALL 24 classes
            true_cat = categorical(trueLabels, allClassNames);
            pred_cat = categorical(predictions, allClassNames);
            
            % Create confusion matrix figure
            fig = figure('Name', 'Confusion Matrix - ASL Letter Recognition', ...
                        'NumberTitle', 'off', ...
                        'Position', [100, 100, 1200, 900]);
            
            % Create confusion chart with ALL 24 classes
            cm = confusionchart(true_cat, pred_cat);
            
            % Customize the chart
            cm.Title = sprintf('ASL Letter Recognition - %d×%d Confusion Matrix', ...
                length(allClassNames), length(allClassNames));
            cm.RowSummary = 'row-normalized';
            cm.ColumnSummary = 'column-normalized';
            cm.FontSize = 10;
                       
            % Improve visualization
            colormap(jet);
            
            % Add text annotations for clarity
            annotation('textbox', [0.1, 0.02, 0.8, 0.05], ...
                'String', 'Rows: True Labels | Columns: Predicted Labels | Diagonal: Correct Predictions', ...
                'EdgeColor', 'none', ...
                'FontSize', 10, ...
                'HorizontalAlignment', 'center');
            
            % Print matrix values for debugging
            fprintf('\n=== Confusion Matrix Values ===\n');
            fprintf('Matrix size: %d×%d\n', size(app.confusionMat));
            fprintf('Classes: %s\n', strjoin(allClassNames, ', '));
            fprintf('\n');
            
            % Display confusion matrix in command window
            disp('Confusion Matrix (True × Predicted):');
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
                fprintf('... (showing first 10×10, full matrix is %d×%d)\n', ...
                    length(allClassNames), length(allClassNames));
            end
        end

        % Selection changed function: TestInputFormatButtonGroup
        function TestInputFormatButtonGroupSelectionChanged(app, event)
            selectedButton = app.TestInputFormatButtonGroup.SelectedObject;
    
            if selectedButton == app.VideoCSVButton
                % Show video + CSV controls
                app.UploadVideoButton.Visible = 'on';
                app.UploadCSVButton.Visible = 'on';
                app.CSVFileLabel.Visible = 'on';
                
                % Hide folder controls
                app.SelectImageFolderButton.Visible = 'off';
                app.ExpectedLetterDropDown.Visible = 'off';
                app.ExpectedLetterDropDown_2Label.Visible = 'off';
                app.FoundFilesListBox.Visible = 'off';
                
            elseif selectedButton == app.FolderPairsButton
                % Show folder selection & listbox
                app.SelectImageFolderButton.Visible = 'on';
                app.SelectImageFolderButton.Text = 'Select Data Folder';
                app.FoundFilesListBox.Visible = 'on';
                
                % Hide others
                app.UploadVideoButton.Visible = 'off';
                app.UploadCSVButton.Visible = 'off';
                app.CSVFileLabel.Visible = 'off';
                app.ExpectedLetterDropDown.Visible = 'off';
                app.ExpectedLetterDropDown_2Label.Visible = 'off';

            else % ImageFolderButton
                % Show image folder controls
                app.SelectImageFolderButton.Visible = 'on';
                app.SelectImageFolderButton.Text = 'Select Image Folder';
                app.ExpectedLetterDropDown.Visible = 'on';
                app.ExpectedLetterDropDown_2Label.Visible = 'on';
                
                % Hide others
                app.UploadVideoButton.Visible = 'off';
                app.UploadCSVButton.Visible = 'off';
                app.CSVFileLabel.Visible = 'off';
                app.FoundFilesListBox.Visible = 'off';
            end
        end

        % Button pushed function: UploadVideoButton
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

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 540];
            app.UIFigure.Name = 'ASL Data Tester';

            % Main Grid Layout
            mainGrid = uigridlayout(app.UIFigure);
            mainGrid.ColumnWidth = {'1.5x', '1x'}; 
            mainGrid.RowHeight = {'1x', 30};

            % -- Input Panel --
            app.InputPanel = uipanel(mainGrid);
            app.InputPanel.Title = 'Input Configuration';
            app.InputPanel.Layout.Row = 1;
            app.InputPanel.Layout.Column = 1;

            % Input Grid Layout
            inputGrid = uigridlayout(app.InputPanel);
            inputGrid.ColumnWidth = {'1x', '1x'};
            % Rows: 1:Mode, 2:Video/Folder, 3:CSV/Letter, 4:Listbox, 5:LoadNet, 6:Eval
            inputGrid.RowHeight = {80, 40, 40, '1x', 40, 50}; 

            % 1. Input Format Group (Absolute within Grid Cell)
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

            % 2. File Selection Row (Overlapped cells)
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
            app.FoundFilesListBox.Layout.Row = 4;
            app.FoundFilesListBox.Layout.Column = [1 2];
            app.FoundFilesListBox.Visible = 'off';

            % 5. Load Network
            app.LoadNetworkButton = uibutton(inputGrid, 'push');
            app.LoadNetworkButton.Text = 'Load Network';
            app.LoadNetworkButton.Layout.Row = 5;
            app.LoadNetworkButton.Layout.Column = [1 2];
            app.LoadNetworkButton.ButtonPushedFcn = createCallbackFcn(app, @LoadNetworkButtonPushed, true);

            % 6. Run Evaluation
            app.RunEvaluationButton = uibutton(inputGrid, 'push');
            app.RunEvaluationButton.Text = 'Run Evaluation';
            app.RunEvaluationButton.FontSize = 18;
            app.RunEvaluationButton.Layout.Row = 6;
            app.RunEvaluationButton.Layout.Column = [1 2];
            app.RunEvaluationButton.ButtonPushedFcn = createCallbackFcn(app, @RunEvaluationButtonPushed, true);

            % -- Results Panel --
            app.ResultsPanel = uipanel(mainGrid);
            app.ResultsPanel.Title = 'Results';
            app.ResultsPanel.Layout.Row = 1;
            app.ResultsPanel.Layout.Column = 2;

            resultsGrid = uigridlayout(app.ResultsPanel);
            resultsGrid.ColumnWidth = {'1x'};
            resultsGrid.RowHeight = {30, 30, 30, 30, 40, '1x'};

            app.AccuracyLabel = uilabel(resultsGrid);
            app.AccuracyLabel.Text = 'Accuracy:';
            app.AccuracyLabel.Layout.Row = 1;

            app.PrecisionLabel = uilabel(resultsGrid);
            app.PrecisionLabel.Text = 'Precision:';
            app.PrecisionLabel.Layout.Row = 2;

            app.RecallLabel = uilabel(resultsGrid);
            app.RecallLabel.Text = 'Recall:';
            app.RecallLabel.Layout.Row = 3;

            app.F1ScoreLabel = uilabel(resultsGrid);
            app.F1ScoreLabel.Text = 'F1-Score:';
            app.F1ScoreLabel.Layout.Row = 4;

            app.ShowConfusionMatrixButton = uibutton(resultsGrid, 'push');
            app.ShowConfusionMatrixButton.Text = 'Show Confusion Matrix';
            app.ShowConfusionMatrixButton.Layout.Row = 5;
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

    % App creation and deletion
    methods (Access = public)

        % Construct app
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

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end