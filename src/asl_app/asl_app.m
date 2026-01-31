classdef asl_app < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure        matlab.ui.Figure

        % --- Toolbar ---
        Toolbar         matlab.ui.container.Toolbar
        LoadNetTool     matlab.ui.container.toolbar.PushTool

        % --- Left Panel (Chart) ---
        ChartPanel      matlab.ui.container.Panel
        ChartImage      matlab.ui.control.Image

        % --- Middle Panel (Camera & Sentence) ---
        CamPanel        matlab.ui.container.Panel
        ImageAxes       matlab.ui.control.UIAxes
        SentenceLabel   matlab.ui.control.Label
        SpaceButton     matlab.ui.control.Button
        DeleteButton    matlab.ui.control.Button

        % --- Right Panel (Results & Controls) ---
        ResultsPanel     matlab.ui.container.Panel
        CurrentCharLabel matlab.ui.control.Label
        ConfGauge        matlab.ui.control.LinearGauge
        GaugeLabel       matlab.ui.control.Label
        Top5Label        matlab.ui.control.Label
        Top5List         matlab.ui.control.Label
        StatusLabel      matlab.ui.control.Label
        StopButton       matlab.ui.control.Button
    end

    properties (Access = private)
        Cam             % Webcam object
        Net             % The trained AI network (SeriesNetwork or DAGNetwork)
        NetInputSize    double = []   % cached [H W]
        NetDisplayName  string = ""   % selected file name
        IsRunning       % Loop flag

        % Properties for "Lock-in" logic
        SentenceText    string  % Accumulates the formed sentence
        LastLockedChar  char    % The last character that was locked in

        % Prediction timing
        LastPredictionTime  uint64  % Timer for prediction interval
        LastPredictedChar   char    % Cached prediction result
        LastPredictedScore  double  % Cached confidence score
        LastScores          double  % Cached full scores array
        LastScoreDiff       double  % Difference between top two predictions
    end

    methods (Access = private)

        % =========================
        % STARTUP
        % =========================
        function startupFcn(app)
            app.StatusLabel.Text = 'Initialising...';
            title(app.ImageAxes, 'Starting camera...');
            drawnow;

            % Initialise state variables
            app.SentenceText = "";
            app.LastLockedChar = '';

            % Initialise prediction timing
            app.LastPredictionTime = tic;
            app.LastPredictedChar = '-';
            app.LastPredictedScore = 0;
            app.LastScores = [];
            app.LastScoreDiff = 0;

            % Do not load a fixed network file anymore
            app.Net = [];
            app.NetInputSize = [];
            app.NetDisplayName = "";

            % Start camera
            try
                app.Cam = webcam;
            catch
                uialert(app.UIFigure, 'No webcam found. Connect one and restart.', 'Camera Error');
                return;
            end

            title(app.ImageAxes, 'Load a network using the toolbar button.');
            app.StatusLabel.Text = 'Camera ready. Load a network to start recognition.';
            app.IsRunning = true;

            % Start loop
            app.recognitionLoop();
        end

        % =========================
        % TOOLBAR CALLBACK
        % =========================
        function LoadNetToolClicked(app)
            [file, path] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select a trained network');
            if isequal(file, 0)
                return;
            end

            fullPath = fullfile(path, file);

            try
                data = load(fullPath);
                [net, ~] = app.findNetworkInLoadedStruct(data);

                if ~(isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork'))
                    error('Unsupported network type. Save a SeriesNetwork or DAGNetwork.');
                end

                app.Net = net;
                app.NetInputSize = app.getNetInputSize(net);
                app.NetDisplayName = string(file);

                % Reset state when loading a new network
                app.LastLockedChar = '';
                app.LastPredictionTime = tic;

                app.StatusLabel.Text = "Network loaded: " + app.NetDisplayName;
                title(app.ImageAxes, '');
                drawnow;

            catch err
                uialert(app.UIFigure, err.message, 'Network Load Error');
            end
        end

        function [net, netVarName] = findNetworkInLoadedStruct(app, data)
            net = [];
            netVarName = "";

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

        function inputSize = getNetInputSize(app, net)
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

        % =========================
        % TOP 5 DISPLAY HELPER
        % =========================
        function updateTop5Display(app, scores, classNames)
            [sortedScores, sortIdx] = sort(scores, 'descend');
            top5Scores = sortedScores(1:min(5, numel(sortedScores)));
            top5Names = classNames(sortIdx(1:min(5, numel(sortIdx))));

            displayLines = strings(5, 1);
            for i = 1:numel(top5Scores)
                pct = top5Scores(i) * 100;
                displayLines(i) = sprintf('%s  -  %.1f%%', char(top5Names(i)), pct);
            end

            app.Top5List.Text = strjoin(displayLines, newline);
        end

        function clearTop5Display(app)
            app.Top5List.Text = '-';
        end

        % =========================
        % MAIN LOOP
        % =========================
        function recognitionLoop(app)

            targetBoxSize = 450;

            % Logic constants
            PREDICTION_INTERVAL = 1.0;
            MIN_CONFIDENCE = 0.6;
            MIN_SCORE_DIFF = 0.20;

            defaultPanelColor = app.CamPanel.BackgroundColor;

            while app.IsRunning && isvalid(app.UIFigure)
                try
                    % --- A. Capture ---
                    img = snapshot(app.Cam);
                    img = fliplr(img);
                    [h, w, ~] = size(img);

                    boxSize = min([targetBoxSize, h, w]);
                    x = round((w - boxSize)/2);
                    if x < 1, x = 1; end
                    y = round((h - boxSize)/2);
                    if y < 1, y = 1; end
                    rect = [x, y, boxSize-1, boxSize-1];

                    % If no network is loaded, show camera feed with guide box
                    if isempty(app.Net) || isempty(app.NetInputSize)
                        imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 4, 'Color', 'white');
                        image(app.ImageAxes, imgDisplay);
                        app.ImageAxes.XTick = [];
                        app.ImageAxes.YTick = [];
                        app.CurrentCharLabel.Text = '-';
                        app.ConfGauge.Value = 0;
                        app.clearTop5Display();
                        app.StatusLabel.Text = 'Load a network from the toolbar to begin.';
                        drawnow limitrate;
                        continue;
                    end

                    % --- B. Classification (only every PREDICTION_INTERVAL seconds) ---
                    timeSincePrediction = toc(app.LastPredictionTime);
                    newPrediction = false;

                    if timeSincePrediction >= PREDICTION_INTERVAL
                        imgHand = imcrop(img, rect);
                        imgResized = imresize(imgHand, app.NetInputSize);
                        [labelCat, scores] = classify(app.Net, imgResized);

                        % Sort scores to get top two
                        sortedScores = sort(scores, 'descend');
                        topScore = sortedScores(1);
                        secondScore = sortedScores(2);

                        app.LastPredictedScore = topScore;
                        app.LastPredictedChar = char(labelCat);
                        app.LastScores = scores;
                        app.LastScoreDiff = topScore - secondScore;
                        app.LastPredictionTime = tic;
                        newPrediction = true;

                        % Update top 5 display
                        classNames = app.Net.Layers(end).Classes;
                        app.updateTop5Display(scores, classNames);
                    end

                    % Use cached values
                    maxScore = app.LastPredictedScore;
                    currentChar = app.LastPredictedChar;
                    scoreDiff = app.LastScoreDiff;

                    % Check if lock-in conditions are met
                    meetsThreshold = (maxScore >= MIN_CONFIDENCE) && (scoreDiff >= MIN_SCORE_DIFF);
                    isDifferentChar = ~strcmp(currentChar, app.LastLockedChar);
                    canLockIn = meetsThreshold && isDifferentChar;

                    % --- C. Automatic lock-in ---
                    boxColor = 'yellow';
                    app.CamPanel.BackgroundColor = defaultPanelColor;

                    if newPrediction && canLockIn && ~isempty(currentChar) && currentChar ~= '-'
                        % Automatically lock in the letter
                        app.appendToSentence(currentChar);
                        app.LastLockedChar = currentChar;
                        boxColor = 'green';
                        app.CamPanel.BackgroundColor = [0.6 1 0.6];
                        app.StatusLabel.Text = ['Locked: ' currentChar];
                    else
                        % Update status based on current state
                        if meetsThreshold && ~isDifferentChar
                            boxColor = 'cyan';
                            app.StatusLabel.Text = sprintf('Same as last (%s)', app.LastLockedChar);
                        elseif meetsThreshold
                            boxColor = 'cyan';
                            app.StatusLabel.Text = 'Ready...';
                        elseif maxScore < MIN_CONFIDENCE
                            boxColor = 'red';
                            app.StatusLabel.Text = sprintf('Conf: %.0f%% (need 40%%)', maxScore * 100);
                        else
                            boxColor = 'yellow';
                            app.StatusLabel.Text = sprintf('Gap: %.0f%% (need 10%%)', scoreDiff * 100);
                        end
                    end

                    % --- D. Update UI ---
                    imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 4, 'Color', boxColor);
                    image(app.ImageAxes, imgDisplay);
                    app.ImageAxes.XTick = [];
                    app.ImageAxes.YTick = [];

                    app.CurrentCharLabel.Text = currentChar;

                    if canLockIn
                        app.CurrentCharLabel.FontColor = [0 0.5 0];
                    else
                        app.CurrentCharLabel.FontColor = [0.85 0.33 0.1];
                    end

                    app.ConfGauge.Value = maxScore * 100;

                    drawnow limitrate;

                catch err
                    app.StatusLabel.Text = ['Error: ' err.message];
                    fprintf('Loop Error: %s\n', err.message);
                    pause(0.5);
                end
            end
        end

        % =========================
        % SENTENCE HELPERS
        % =========================
        function appendToSentence(app, charToAdd)
            app.SentenceText = app.SentenceText + charToAdd;
            app.SentenceLabel.Text = app.SentenceText;
        end

        % =========================
        % UI CALLBACKS
        % =========================
        function KeyPressed(app, event)
            switch event.Key
                case 'space'
                    app.appendToSentence(" ");
                case 'backspace'
                    app.DeleteButtonPushed();
            end
        end

        function SpaceButtonPushed(app, ~)
            app.appendToSentence(" ");
        end

        function DeleteButtonPushed(app, ~)
            currentTxt = char(app.SentenceText);
            if ~isempty(currentTxt)
                app.SentenceText = string(currentTxt(1:end-1));
                app.SentenceLabel.Text = app.SentenceText;
            end
        end

        function StopButtonPushed(app, ~)
            app.IsRunning = false;
            pause(0.1);
            if ~isempty(app.Cam)
                delete(app.Cam);
                app.Cam = [];
            end
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end

        function UIFigureCloseRequest(app, ~)
            app.StopButtonPushed();
        end
    end

    methods (Access = public)
        % =========================
        % CREATE COMPONENTS
        % =========================
        function createComponents(app)
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1100 650];
            app.UIFigure.Name = 'ASL Pro Translator v3';
            app.UIFigure.Color = [0.92 0.93 0.94];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);
            app.UIFigure.KeyPressFcn = createCallbackFcn(app, @KeyPressed, true);

            % === TOOLBAR ===
            app.Toolbar = uitoolbar(app.UIFigure);

            icon = zeros(16,16,3);
            icon(:,:,2) = 0.6;
            icon(3:14,3:14,:) = 0.9;

            app.LoadNetTool = uipushtool(app.Toolbar, ...
                'Tooltip', 'Load Network (.mat)', ...
                'CData', icon, ...
                'ClickedCallback', @(src,event)app.LoadNetToolClicked());

            % Define column geometry
            col1_x = 20;  col1_w = 320;
            col2_x = 360; col2_w = 480;
            col3_x = 860; col3_w = 220;
            base_y = 20; panel_h = 610;

            % === LEFT PANEL: Reference chart ===
            app.ChartPanel = uipanel(app.UIFigure);
            app.ChartPanel.Title = 'Reference Chart';
            app.ChartPanel.Position = [col1_x base_y col1_w panel_h];
            app.ChartPanel.BackgroundColor = 'white';

            imagePath = 'asl_chart.jpeg';
            if ~exist(imagePath, 'file')
                imwrite(zeros(300,300,3)+0.9, 'asl_chart_placeholder.jpeg');
                imagePath = 'asl_chart_placeholder.jpeg';
            end

            app.ChartImage = uiimage(app.ChartPanel);
            app.ChartImage.Position = [10 50 300 500];
            app.ChartImage.ImageSource = imagePath;
            app.ChartImage.ScaleMethod = 'fit';

            % === MIDDLE PANEL: Camera & sentence ===
            app.CamPanel = uipanel(app.UIFigure);
            app.CamPanel.Title = 'Live Input & Translation';
            app.CamPanel.Position = [col2_x base_y col2_w panel_h];

            app.ImageAxes = uiaxes(app.CamPanel);
            app.ImageAxes.Position = [15 180 450 400];
            app.ImageAxes.XTick = [];
            app.ImageAxes.YTick = [];
            app.ImageAxes.Box = 'on';
            app.ImageAxes.BackgroundColor = 'black';

            app.SentenceLabel = uilabel(app.CamPanel);
            app.SentenceLabel.Position = [15 80 450 80];
            app.SentenceLabel.Text = '';
            app.SentenceLabel.BackgroundColor = 'white';
            app.SentenceLabel.FontSize = 24;
            app.SentenceLabel.FontWeight = 'bold';
            app.SentenceLabel.VerticalAlignment = 'top';
            app.SentenceLabel.WordWrap = 'on';

            app.SpaceButton = uibutton(app.CamPanel, 'push');
            app.SpaceButton.Position = [15 20 220 50];
            app.SpaceButton.Text = 'SPACE [ _ ]';
            app.SpaceButton.FontSize = 16;
            app.SpaceButton.ButtonPushedFcn = createCallbackFcn(app, @SpaceButtonPushed, true);

            app.DeleteButton = uibutton(app.CamPanel, 'push');
            app.DeleteButton.Position = [245 20 220 50];
            app.DeleteButton.Text = 'DELETE [ Backspace ]';
            app.DeleteButton.FontSize = 16;
            app.DeleteButton.ButtonPushedFcn = createCallbackFcn(app, @DeleteButtonPushed, true);

            % === RIGHT PANEL: Results & controls ===
            app.ResultsPanel = uipanel(app.UIFigure);
            app.ResultsPanel.Title = 'Real-Time Analysis';
            app.ResultsPanel.Position = [col3_x base_y col3_w panel_h];

            app.CurrentCharLabel = uilabel(app.ResultsPanel);
            app.CurrentCharLabel.Position = [10 420 200 120];
            app.CurrentCharLabel.Text = '-';
            app.CurrentCharLabel.FontSize = 90;
            app.CurrentCharLabel.FontWeight = 'bold';
            app.CurrentCharLabel.HorizontalAlignment = 'center';

            app.ConfGauge = uigauge(app.ResultsPanel, 'linear');
            app.ConfGauge.Position = [10 380 200 40];
            app.ConfGauge.Limits = [0 100];
            app.ConfGauge.ScaleColors = [0.8 0 0; 1 0.8 0; 0 0.6 0];
            app.ConfGauge.ScaleColorLimits = [0 40; 40 70; 70 100];

            app.GaugeLabel = uilabel(app.ResultsPanel);
            app.GaugeLabel.Position = [10 355 200 22];
            app.GaugeLabel.Text = 'Confidence %';
            app.GaugeLabel.HorizontalAlignment = 'center';

            app.Top5Label = uilabel(app.ResultsPanel);
            app.Top5Label.Position = [10 310 200 22];
            app.Top5Label.Text = 'Top 5 Predictions';
            app.Top5Label.FontWeight = 'bold';
            app.Top5Label.HorizontalAlignment = 'center';

            app.Top5List = uilabel(app.ResultsPanel);
            app.Top5List.Position = [10 190 200 120];
            app.Top5List.Text = '-';
            app.Top5List.FontSize = 14;
            app.Top5List.FontName = 'Consolas';
            app.Top5List.VerticalAlignment = 'top';
            app.Top5List.BackgroundColor = [0.95 0.95 0.95];

            app.StatusLabel = uilabel(app.ResultsPanel);
            app.StatusLabel.Position = [10 120 200 60];
            app.StatusLabel.Text = 'Scanning...';
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.WordWrap = 'on';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.FontAngle = 'italic';

            app.StopButton = uibutton(app.ResultsPanel, 'push');
            app.StopButton.Position = [10 20 200 60];
            app.StopButton.Text = 'STOP SYSTEM';
            app.StopButton.ButtonPushedFcn = createCallbackFcn(app, @StopButtonPushed, true);
            app.StopButton.BackgroundColor = [0.7 0.2 0.2];
            app.StopButton.FontColor = 'white';
            app.StopButton.FontSize = 16;
            app.StopButton.FontWeight = 'bold';

            app.UIFigure.Visible = 'on';
        end
    end

    methods (Access = public)
        % =========================
        % CONSTRUCTOR
        % =========================
        function app = asl_app
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
        end

        function delete(app)
            if ~isempty(app.Cam)
                try
                    delete(app.Cam);
                catch
                end
                app.Cam = [];
            end
            if ~isempty(app.UIFigure) && isvalid(app.UIFigure)
                delete(app.UIFigure)
            end
        end
    end
end