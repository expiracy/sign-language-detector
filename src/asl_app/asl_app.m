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
        LastStableChar  char    % The character currently being tracked
        StableStartTime uint64  % Timer for stability check (tic handle)
        LastLockInTime  uint64  % Timer for the 1s cooldown after a guess (tic handle)
    end

    methods (Access = private)

        % =========================
        % STARTUP
        % =========================
        function startupFcn(app)
            app.StatusLabel.Text = 'Initialising...';
            title(app.ImageAxes, 'Starting camera...');
            drawnow;

            % 1. Initialise state variables
            app.SentenceText = "";
            app.LastStableChar = '';
            app.StableStartTime = tic;
            app.LastLockInTime = tic - 2; % 2 seconds in the past

            % 2. Do not load a fixed network file anymore
            app.Net = [];
            app.NetInputSize = [];
            app.NetDisplayName = "";

            % 3. Start camera
            try
                app.Cam = webcam;
            catch
                uialert(app.UIFigure, 'No webcam found. Connect one and restart.', 'Camera Error');
                return;
            end

            title(app.ImageAxes, 'Load a network using the toolbar button.');
            app.StatusLabel.Text = 'Camera ready. Load a network to start recognition.';
            app.IsRunning = true;

            % 4. Start loop
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

                % Only support SeriesNetwork / DAGNetwork because we use classify(...)
                if ~(isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork'))
                    error('Unsupported network type. Save a SeriesNetwork or DAGNetwork.');
                end

                app.Net = net;
                app.NetInputSize = app.getNetInputSize(net);
                app.NetDisplayName = string(file);

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
        % MAIN LOOP
        % =========================
        function recognitionLoop(app)

            % Increased box size
            targetBoxSize = 450;

            % Logic constants
            CONF_THRESHOLD = 0.80; % 80% confidence required
            TIME_THRESHOLD = 0.5;  % Must hold for 0.5s
            COOLDOWN_TIME  = 1.0;  % Pause for 1s after lock-in

            % Default panel colour
            defaultPanelColor = app.CamPanel.BackgroundColor;

            while app.IsRunning && isvalid(app.UIFigure)
                try
                    % --- A. Capture ---
                    img = snapshot(app.Cam);
                    img = fliplr(img);
                    [h, w, ~] = size(img);

                    % If no network is loaded, just show the camera feed
                    if isempty(app.Net) || isempty(app.NetInputSize)
                        image(app.ImageAxes, img);
                        app.ImageAxes.XTick = []; app.ImageAxes.YTick = [];
                        app.CurrentCharLabel.Text = '-';
                        app.ConfGauge.Value = 0;
                        app.StatusLabel.Text = 'Load a network from the toolbar to begin.';
                        drawnow limitrate;
                        continue;
                    end

                    % Calculate centre crop
                    boxSize = min([targetBoxSize, h, w]);
                    x = round((w - boxSize)/2); if x < 1, x = 1; end
                    y = round((h - boxSize)/2); if y < 1, y = 1; end
                    rect = [x, y, boxSize-1, boxSize-1];

                    % --- B. Cooldown check ---
                    timeSinceLock = toc(app.LastLockInTime);

                    if timeSinceLock < COOLDOWN_TIME
                        % In cooldown, show green box and skip classification
                        boxColor = 'green';

                        imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 6, 'Color', boxColor);
                        image(app.ImageAxes, imgDisplay);
                        app.ImageAxes.XTick = []; app.ImageAxes.YTick = [];

                        if timeSinceLock < 0.3
                            app.CamPanel.BackgroundColor = [0.6 1 0.6]; % light green
                        else
                            app.CamPanel.BackgroundColor = defaultPanelColor;
                        end

                        drawnow limitrate;
                        continue;
                    else
                        app.CamPanel.BackgroundColor = defaultPanelColor;
                    end

                    % --- C. Classification ---
                    imgHand = imcrop(img, rect);
                    imgResized = imresize(imgHand, app.NetInputSize);
                    [labelCat, scores] = classify(app.Net, imgResized);

                    maxScore = max(scores);
                    currentChar = char(labelCat);

                    % --- D. Lock-in logic ---
                    boxColor = 'yellow';

                    if maxScore > CONF_THRESHOLD
                        boxColor = 'cyan';

                        if strcmp(currentChar, app.LastStableChar)
                            if toc(app.StableStartTime) > TIME_THRESHOLD
                                % Lock in
                                app.appendToSentence(currentChar);

                                % Start cooldown
                                app.LastLockInTime = tic;

                                % Reset stability
                                app.StableStartTime = tic;

                                boxColor = 'green';
                                app.StatusLabel.Text = ['Locked: ' currentChar];
                            end
                        else
                            app.LastStableChar = currentChar;
                            app.StableStartTime = tic;
                        end
                    else
                        app.LastStableChar = '';
                        app.StableStartTime = tic;
                        if maxScore < 0.5
                            boxColor = 'red';
                        end
                        app.StatusLabel.Text = 'Scanning...';
                    end

                    % --- E. Update UI ---
                    imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 4, 'Color', boxColor);
                    image(app.ImageAxes, imgDisplay);
                    app.ImageAxes.XTick = []; app.ImageAxes.YTick = [];

                    app.CurrentCharLabel.Text = currentChar;

                    if maxScore < CONF_THRESHOLD
                        app.CurrentCharLabel.FontColor = [0.85 0.33 0.1]; % orange
                    else
                        app.CurrentCharLabel.FontColor = [0 0.5 0]; % green
                    end

                    app.ConfGauge.Value = maxScore * 100;

                    drawnow limitrate;

                catch err
                    fprintf('Loop Error: %s\n', err.message);
                    app.IsRunning = false;
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
        function SpaceButtonPushed(app, event)
            app.appendToSentence(" ");
        end

        function DeleteButtonPushed(app, event)
            currentTxt = char(app.SentenceText);
            if ~isempty(currentTxt)
                app.SentenceText = string(currentTxt(1:end-1));
                app.SentenceLabel.Text = app.SentenceText;
            end
        end

        function StopButtonPushed(app, event)
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

        function UIFigureCloseRequest(app, event)
            app.StopButtonPushed();
        end
    end

    methods (Access = public)
        % =========================
        % CREATE COMPONENTS
        % =========================
        function createComponents(app)
            % Main window style
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1100 650];
            app.UIFigure.Name = 'ASL Pro Translator v3';
            app.UIFigure.Color = [0.92 0.93 0.94];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % === TOOLBAR ===
            app.Toolbar = uitoolbar(app.UIFigure);

            % Simple 16x16 icon (green square)
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

            % Check for image file
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
            app.ImageAxes.XTick = []; app.ImageAxes.YTick = [];
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
            app.DeleteButton.Text = 'DELETE [ <- ]';
            app.DeleteButton.FontSize = 16;
            app.DeleteButton.ButtonPushedFcn = createCallbackFcn(app, @DeleteButtonPushed, true);

            % === RIGHT PANEL: Results & controls ===
            app.ResultsPanel = uipanel(app.UIFigure);
            app.ResultsPanel.Title = 'Real-Time Analysis';
            app.ResultsPanel.Position = [col3_x base_y col3_w panel_h];

            app.CurrentCharLabel = uilabel(app.ResultsPanel);
            app.CurrentCharLabel.Position = [10 350 200 180];
            app.CurrentCharLabel.Text = '-';
            app.CurrentCharLabel.FontSize = 120;
            app.CurrentCharLabel.FontWeight = 'bold';
            app.CurrentCharLabel.HorizontalAlignment = 'center';

            app.ConfGauge = uigauge(app.ResultsPanel, 'linear');
            app.ConfGauge.Position = [10 300 200 40];
            app.ConfGauge.Limits = [0 100];
            app.ConfGauge.ScaleColors = [0.8 0 0; 1 0.8 0; 0 0.6 0];
            app.ConfGauge.ScaleColorLimits = [0 60; 60 80; 80 100];

            app.GaugeLabel = uilabel(app.ResultsPanel);
            app.GaugeLabel.Position = [10 275 200 22];
            app.GaugeLabel.Text = 'Confidence %';
            app.GaugeLabel.HorizontalAlignment = 'center';

            app.StatusLabel = uilabel(app.ResultsPanel);
            app.StatusLabel.Position = [10 150 200 100];
            app.StatusLabel.Text = 'Hold sign steady for 0.5s to lock it in.';
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.WordWrap = 'on';
            app.StatusLabel.FontSize = 14;
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
                try, delete(app.Cam); catch, end
                app.Cam = [];
            end
            if ~isempty(app.UIFigure) && isvalid(app.UIFigure)
                delete(app.UIFigure)
            end
        end
    end
end

