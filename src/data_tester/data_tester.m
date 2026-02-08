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
    end

    
    properties (Access = private)

        % Data storage
        videoPath
        csvPath
        imageFolderPath
        modelPath

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

        function loadVideoData(app)
            try
                % Read CSV
                data = readtable(app.csvPath);
                
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
                if ~isfile(app.videoPath)
                    error('Video file not found: %s', app.videoPath);
                end
                vid = VideoReader(app.videoPath);
                
                app.frames = {};
                app.trueLabels = {};
                
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
                
                app.StatusLabel.Text = sprintf('Loaded %d frames from %d letters', ...
                    length(app.frames), height(data));
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
            
            % Set initial visibility
            app.UploadVideoButton.Visible = 'on';
            app.UploadCSVButton.Visible = 'on';
            app.CSVFileLabel.Visible = 'on';
            
            app.SelectImageFolderButton.Visible = 'off';
            app.ExpectedLetterDropDown.Visible = 'off';
            
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
            folder = uigetdir(pwd, 'Select Image Folder');
            if folder ~= 0
                app.imageFolderPath = folder;

                app.FileLabel.Text = folder;
                app.SelectImageFolderButton.FontColor = [0 1 0];
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
                % Show video + CSV controls, hide image folder controls
                app.UploadVideoButton.Visible = 'on';
                app.UploadCSVButton.Visible = 'on';
                app.CSVFileLabel.Visible = 'on';
                
                app.SelectImageFolderButton.Visible = 'off';
                app.ExpectedLetterDropDown.Visible = 'off';
                
            else % ImageFolderRadioButton
                % Show image folder controls, hide video + CSV controls
                app.UploadVideoButton.Visible = 'off';
                app.UploadCSVButton.Visible = 'off';
                app.CSVFileLabel.Visible = 'off';
                
                app.SelectImageFolderButton.Visible = 'on';
                app.ExpectedLetterDropDown.Visible = 'on';
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
            app.UIFigure.Position = [100 100 402 339];
            app.UIFigure.Name = 'MATLAB App';

            % Create InputPanel
            app.InputPanel = uipanel(app.UIFigure);
            app.InputPanel.Title = 'Input';
            app.InputPanel.Position = [13 36 215 295];

            % Create TestInputFormatButtonGroup
            app.TestInputFormatButtonGroup = uibuttongroup(app.InputPanel);
            app.TestInputFormatButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @TestInputFormatButtonGroupSelectionChanged, true);
            app.TestInputFormatButtonGroup.TitlePosition = 'centertop';
            app.TestInputFormatButtonGroup.Title = 'Test Input Format';
            app.TestInputFormatButtonGroup.Position = [8 195 123 70];

            % Create VideoCSVButton
            app.VideoCSVButton = uiradiobutton(app.TestInputFormatButtonGroup);
            app.VideoCSVButton.Text = 'Video + CSV';
            app.VideoCSVButton.Position = [11 24 91 22];
            app.VideoCSVButton.Value = true;

            % Create ImageFolderButton
            app.ImageFolderButton = uiradiobutton(app.TestInputFormatButtonGroup);
            app.ImageFolderButton.Text = 'Image Folder';
            app.ImageFolderButton.Position = [11 2 93 22];

            % Create UploadVideoButton
            app.UploadVideoButton = uibutton(app.InputPanel, 'push');
            app.UploadVideoButton.ButtonPushedFcn = createCallbackFcn(app, @UploadVideoButtonPushed, true);
            app.UploadVideoButton.Position = [10 166 100 23];
            app.UploadVideoButton.Text = 'Upload Video';

            % Create SelectImageFolderButton
            app.SelectImageFolderButton = uibutton(app.InputPanel, 'push');
            app.SelectImageFolderButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImageFolderButtonPushed, true);
            app.SelectImageFolderButton.Position = [10 166 122 23];
            app.SelectImageFolderButton.Text = 'Select Image Folder';

            % Create LoadNetworkButton
            app.LoadNetworkButton = uibutton(app.InputPanel, 'push');
            app.LoadNetworkButton.ButtonPushedFcn = createCallbackFcn(app, @LoadNetworkButtonPushed, true);
            app.LoadNetworkButton.Position = [8 62 100 23];
            app.LoadNetworkButton.Text = 'Load Network';

            % Create RunEvaluationButton
            app.RunEvaluationButton = uibutton(app.InputPanel, 'push');
            app.RunEvaluationButton.ButtonPushedFcn = createCallbackFcn(app, @RunEvaluationButtonPushed, true);
            app.RunEvaluationButton.FontSize = 18;
            app.RunEvaluationButton.Position = [8 19 153 39];
            app.RunEvaluationButton.Text = 'Run Evaluation';

            % Create FileLabel
            app.FileLabel = uilabel(app.InputPanel);
            app.FileLabel.Position = [29 145 186 22];
            app.FileLabel.Text = '';

            % Create ExpectedLetterDropDown_2Label
            app.ExpectedLetterDropDown_2Label = uilabel(app.InputPanel);
            app.ExpectedLetterDropDown_2Label.HorizontalAlignment = 'right';
            app.ExpectedLetterDropDown_2Label.Position = [8 118 89 22];
            app.ExpectedLetterDropDown_2Label.Text = 'Expected Letter';

            % Create ExpectedLetterDropDown
            app.ExpectedLetterDropDown = uidropdown(app.InputPanel);
            app.ExpectedLetterDropDown.Items = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'};
            app.ExpectedLetterDropDown.Position = [112 118 49 22];
            app.ExpectedLetterDropDown.Value = 'A';

            % Create UploadCSVButton
            app.UploadCSVButton = uibutton(app.InputPanel, 'push');
            app.UploadCSVButton.ButtonPushedFcn = createCallbackFcn(app, @UploadCSVButtonPushed, true);
            app.UploadCSVButton.Position = [8 117 100 23];
            app.UploadCSVButton.Text = 'Upload CSV';

            % Create CSVFileLabel
            app.CSVFileLabel = uilabel(app.UIFigure);
            app.CSVFileLabel.Position = [43 130 186 22];
            app.CSVFileLabel.Text = '';

            % Create StatusLabel
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.Position = [14 9 487 22];
            app.StatusLabel.Text = 'Note';

            % Create ResultsPanel
            app.ResultsPanel = uipanel(app.UIFigure);
            app.ResultsPanel.Title = 'Results';
            app.ResultsPanel.Position = [228 36 160 295];

            % Create AccuracyLabel
            app.AccuracyLabel = uilabel(app.ResultsPanel);
            app.AccuracyLabel.Position = [9 242 138 22];
            app.AccuracyLabel.Text = 'Accuracy:';

            % Create PrecisionLabel
            app.PrecisionLabel = uilabel(app.ResultsPanel);
            app.PrecisionLabel.Position = [9 220 138 22];
            app.PrecisionLabel.Text = 'Precision:';

            % Create RecallLabel
            app.RecallLabel = uilabel(app.ResultsPanel);
            app.RecallLabel.Position = [9 198 137 22];
            app.RecallLabel.Text = 'Recall:';

            % Create F1ScoreLabel
            app.F1ScoreLabel = uilabel(app.ResultsPanel);
            app.F1ScoreLabel.Position = [9 174 138 22];
            app.F1ScoreLabel.Text = 'F1-Score:';

            % Create ShowConfusionMatrixButton
            app.ShowConfusionMatrixButton = uibutton(app.ResultsPanel, 'push');
            app.ShowConfusionMatrixButton.ButtonPushedFcn = createCallbackFcn(app, @ShowConfusionMatrixButtonPushed, true);
            app.ShowConfusionMatrixButton.Position = [9 136 138 23];
            app.ShowConfusionMatrixButton.Text = 'Show Confusion Matrix';

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