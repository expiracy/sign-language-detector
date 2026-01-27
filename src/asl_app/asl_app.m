classdef asl_app < matlab.apps.AppBase

    properties (Access = public)
        UIFigure        matlab.ui.Figure

        % Toolbar
        Toolbar         matlab.ui.container.Toolbar
        LoadNetTool     matlab.ui.container.toolbar.PushTool

        % Left Panel (Chart)
        ChartPanel      matlab.ui.container.Panel
        ChartImage      matlab.ui.control.Image

        % Middle Panel (Camera & Sentence)
        CamPanel        matlab.ui.container.Panel
        ImageAxes       matlab.ui.control.UIAxes
        SentenceLabel   matlab.ui.control.Label
        SpaceButton     matlab.ui.control.Button
        DeleteButton    matlab.ui.control.Button

        % Right Panel (Results & Controls)
        ResultsPanel     matlab.ui.container.Panel
        CurrentCharLabel matlab.ui.control.Label
        ConfGauge        matlab.ui.control.LinearGauge
        GaugeLabel       matlab.ui.control.Label
        StatusLabel      matlab.ui.control.Label
        StopButton       matlab.ui.control.Button
    end

    properties (Access = private)
        Cam
        Net
        NetInputSize
        NetDisplayName  string = ""
        IsRunning       logical = false

        % Timer (no type constraint to allow empty assignment)
        LoopTimer

        % Drawing handles
        FrameImageHandle
        RectHandle

        % Lock-in logic
        SentenceText    string = ""
        LastStableChar  char = ''
        StableStartTime
        LastLockInTime

        % Settings
        TargetBoxSize   double = 450
        DefaultPanelColor

        % Thresholds
        ConfThreshold   double = 0.80
        TimeThreshold   double = 0.5
        CooldownTime    double = 1.0
    end

    methods (Access = private)

        function startupFcn(app)
            app.StatusLabel.Text = 'Initialising...';
            drawnow;

            % Initialise state
            app.SentenceText = "";
            app.LastStableChar = '';
            app.StableStartTime = tic;
            app.LastLockInTime = tic - 2;

            % Network state
            app.Net = [];
            app.NetInputSize = [];
            app.NetDisplayName = "";

            % Drawing handles
            app.FrameImageHandle = [];
            app.RectHandle = [];

            % Timer
            app.LoopTimer = [];

            % Store default panel colour
            app.DefaultPanelColor = app.CamPanel.BackgroundColor;

            % Start camera
            try
                app.Cam = webcam;
            catch
                uialert(app.UIFigure, 'No webcam found. Connect one and restart.', 'Camera Error');
                return;
            end

            app.StatusLabel.Text = 'Camera ready. Load a network to start recognition.';
            app.IsRunning = true;

            % Start timer
            app.startLoopTimer();
        end

        function startLoopTimer(app)
            app.stopLoopTimer();

            app.LoopTimer = timer( ...
                'ExecutionMode', 'fixedSpacing', ...
                'Period', 0.05, ...
                'BusyMode', 'drop', ...
                'TimerFcn', @(~,~)app.onTimerTick(), ...
                'ErrorFcn', @(~,e)app.onTimerError(e));

            start(app.LoopTimer);
        end

        function stopLoopTimer(app)
            t = app.LoopTimer;
            if ~isempty(t) && isa(t, 'timer') && isvalid(t)
                try stop(t); catch, end
                try delete(t); catch, end
            end
            app.LoopTimer = [];
        end

        function onTimerError(app, e)
            if isfield(e, 'Data') && isfield(e.Data, 'message')
                msg = e.Data.message;
            elseif isprop(e, 'message')
                msg = e.message;
            else
                msg = 'Unknown error';
            end
            fprintf('Timer error: %s\n', msg);
            app.stopLoopTimer();
            app.IsRunning = false;
        end

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

                app.StatusLabel.Text = "Network loaded: " + app.NetDisplayName;
                drawnow;

            catch err
                uialert(app.UIFigure, err.message, 'Network Load Error');
            end
        end

        function [net, netVarName] = findNetworkInLoadedStruct(~, data)
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

        function updateDisplay(app, img, rect, boxColor)
            % Update or create image handle
            if isempty(app.FrameImageHandle) || ~isvalid(app.FrameImageHandle)
                app.FrameImageHandle = image(app.ImageAxes, img);
                app.ImageAxes.XTick = [];
                app.ImageAxes.YTick = [];
                hold(app.ImageAxes, 'on');
            else
                app.FrameImageHandle.CData = img;
            end

            % Update or create rectangle handle
            if isempty(app.RectHandle) || ~isvalid(app.RectHandle)
                app.RectHandle = rectangle(app.ImageAxes, ...
                    'Position', rect, ...
                    'EdgeColor', boxColor, ...
                    'LineWidth', 4);
            else
                app.RectHandle.Position = rect;
                app.RectHandle.EdgeColor = boxColor;
            end
        end

        function onTimerTick(app)
            % Guard
            if ~app.IsRunning || isempty(app.UIFigure) || ~isvalid(app.UIFigure)
                app.stopLoopTimer();
                return;
            end

            % Capture frame
            try
                img = snapshot(app.Cam);
            catch
                app.StatusLabel.Text = 'Camera read failed.';
                return;
            end

            img = fliplr(img);
            [h, w, ~] = size(img);

            % Calculate crop rectangle
            boxSize = min([app.TargetBoxSize, h, w]);
            x = round((w - boxSize) / 2);
            if x < 1, x = 1; end
            y = round((h - boxSize) / 2);
            if y < 1, y = 1; end
            rect = [x, y, boxSize, boxSize];

            % No network loaded
            if isempty(app.Net) || isempty(app.NetInputSize)
                app.updateDisplay(img, rect, [1 1 0]);
                app.CurrentCharLabel.Text = '-';
                app.ConfGauge.Value = 0;
                app.StatusLabel.Text = 'Load a network from the toolbar to begin.';
                drawnow limitrate;
                return;
            end

            % Cooldown check
            timeSinceLock = toc(app.LastLockInTime);

            if timeSinceLock < app.CooldownTime
                app.updateDisplay(img, rect, [0 1 0]);

                if timeSinceLock < 0.3
                    app.CamPanel.BackgroundColor = [0.6 1 0.6];
                else
                    app.CamPanel.BackgroundColor = app.DefaultPanelColor;
                end

                drawnow limitrate;
                return;
            else
                app.CamPanel.BackgroundColor = app.DefaultPanelColor;
            end

            % Classification
            try
                cropRect = [x, y, boxSize-1, boxSize-1];
                imgHand = imcrop(img, cropRect);
                imgResized = imresize(imgHand, app.NetInputSize);

                % Handle grayscale networks
                layers = app.Net.Layers;
                for i = 1:numel(layers)
                    if isprop(layers(i), 'InputSize')
                        sz = layers(i).InputSize;
                        if numel(sz) >= 3 && sz(3) == 1 && size(imgResized, 3) == 3
                            imgResized = rgb2gray(imgResized);
                        end
                        break;
                    end
                end

                [labelCat, scores] = classify(app.Net, imgResized);
                maxScore = max(scores);
                currentChar = char(labelCat);

            catch classifyErr
                fprintf('Classification error: %s\n', classifyErr.message);
                app.updateDisplay(img, rect, [1 0 0]);
                app.StatusLabel.Text = 'Classification error.';
                drawnow limitrate;
                return;
            end

            % Lock-in logic
            boxColor = [1 1 0];

            if maxScore > app.ConfThreshold
                boxColor = [0 1 1];

                if strcmp(currentChar, app.LastStableChar)
                    if toc(app.StableStartTime) > app.TimeThreshold
                        app.appendToSentence(currentChar);
                        app.LastLockInTime = tic;
                        app.StableStartTime = tic;
                        boxColor = [0 1 0];
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
                    boxColor = [1 0 0];
                end
                app.StatusLabel.Text = 'Scanning...';
            end

            % Update display
            app.updateDisplay(img, rect, boxColor);

            app.CurrentCharLabel.Text = currentChar;

            if maxScore < app.ConfThreshold
                app.CurrentCharLabel.FontColor = [0.85 0.33 0.1];
            else
                app.CurrentCharLabel.FontColor = [0 0.5 0];
            end

            app.ConfGauge.Value = maxScore * 100;

            drawnow limitrate;
        end

        function appendToSentence(app, charToAdd)
            app.SentenceText = app.SentenceText + charToAdd;
            app.SentenceLabel.Text = app.SentenceText;
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
            app.stopLoopTimer();
            pause(0.1);

            if ~isempty(app.Cam)
                try delete(app.Cam); catch, end
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

        function createComponents(app)
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1100 650];
            app.UIFigure.Name = 'ASL Pro Translator v3';
            app.UIFigure.Color = [0.92 0.93 0.94];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            % Toolbar
            app.Toolbar = uitoolbar(app.UIFigure);

            icon = zeros(16,16,3);
            icon(:,:,2) = 0.6;
            icon(3:14,3:14,:) = 0.9;

            app.LoadNetTool = uipushtool(app.Toolbar, ...
                'Tooltip', 'Load Network (.mat)', ...
                'CData', icon, ...
                'ClickedCallback', @(~,~)app.LoadNetToolClicked());

            % Layout
            col1_x = 20;  col1_w = 320;
            col2_x = 360; col2_w = 480;
            col3_x = 860; col3_w = 220;
            base_y = 20;  panel_h = 610;

            % Left panel
            app.ChartPanel = uipanel(app.UIFigure);
            app.ChartPanel.Title = 'Reference Chart';
            app.ChartPanel.Position = [col1_x base_y col1_w panel_h];
            app.ChartPanel.BackgroundColor = 'white';

            imagePath = 'asl_chart.jpeg';
            if ~exist(imagePath, 'file')
                imwrite(zeros(300,300,3,'uint8')+230, 'asl_chart_placeholder.jpeg');
                imagePath = 'asl_chart_placeholder.jpeg';
            end

            app.ChartImage = uiimage(app.ChartPanel);
            app.ChartImage.Position = [10 50 300 500];
            app.ChartImage.ImageSource = imagePath;
            app.ChartImage.ScaleMethod = 'fit';

            % Middle panel
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
            app.DeleteButton.Text = 'DELETE [ <- ]';
            app.DeleteButton.FontSize = 16;
            app.DeleteButton.ButtonPushedFcn = createCallbackFcn(app, @DeleteButtonPushed, true);

            % Right panel
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

        function app = asl_app
            createComponents(app)
            registerApp(app, app.UIFigure)
            runStartupFcn(app, @startupFcn)
        end

        function delete(app)
            app.IsRunning = false;
            
            t = app.LoopTimer;
            if ~isempty(t) && isa(t, 'timer') && isvalid(t)
                try stop(t); catch, end
                try delete(t); catch, end
            end

            if ~isempty(app.Cam)
                try delete(app.Cam); catch, end
                app.Cam = [];
            end

            if ~isempty(app.UIFigure) && isvalid(app.UIFigure)
                delete(app.UIFigure)
            end
        end
    end
end