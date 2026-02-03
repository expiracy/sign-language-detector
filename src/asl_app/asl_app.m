classdef asl_app < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure        matlab.ui.Figure

        % --- Toolbar ---
        Toolbar         matlab.ui.container.Toolbar
        LoadNetTool     matlab.ui.container.toolbar.PushTool
        ImageInputSelectionDropdown matlab.ui.container.toolbar.ToggleTool

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
        ConfThreshSlider matlab.ui.control.Slider
        ConfThreshLabel  matlab.ui.control.Label
        GapThreshSlider  matlab.ui.control.Slider
        GapThreshLabel   matlab.ui.control.Label
        StatusLabel      matlab.ui.control.Label
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
        boxColor

        % Prediction timing
        LastPredictionTime  uint64  % Timer for prediction interval
        LastPredictedChar   char    % Cached prediction result
        LastPredictedScore  double  % Cached confidence score
        LastScores          double  % Cached full scores array
        LastScoreDiff       double  % Difference between top two predictions

        % Video Input
        InputVideoReader
        InputVideoLoaded
        LastImgFrameTime

        % Configurable thresholds
        MinConfidence       double = 0.7
        MinScoreDiff        double = 0.3
        targetBoxSize       uint64 = 450
        PREDICTION_INTERVAL double = 1.0
        MAX_CAPTURE_RETRIES uint64 = 3
    end

    methods (Access = private)

        % =========================
        % STARTUP
        % =========================
        function startupFcn(app)
            app.StatusLabel.Text = 'Initialising...';
            title(app.ImageAxes, ' ');

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

            % Start camera with timeout configuration
            app.InputChoiceUpdated()

            app.StatusLabel.Text = 'Camera ready. Load a network to start recognition.';
            app.IsRunning = true;

            app.InputVideoLoaded = false;
            app.LastImgFrameTime = tic();

            % Start loop
            app.recognitionLoop();
        end

        % =========================
        % Image Aquisition Helpers
        % =========================
        function success = reconnectCamera(app)
            try
                % Release existing camera
                if ~isempty(app.Cam)
                    delete(app.Cam);
                    app.Cam = [];
                end
                pause(1.0);
                
                % Attempt to reconnect
                app.Cam = webcam;
                if isprop(app.Cam, 'Timeout')
                    app.Cam.Timeout = 10;
                end
                
                app.StatusLabel.Text = 'Camera reconnected.';
                success = true;
            catch
                app.StatusLabel.Text = 'Camera reconnection failed.';
                success = false;
            end
        end

        function img = getImage(app)
            if strcmp(app.ImageInputSelectionDropdown.State,'on')
                captureSuccess = false;
                
                for attempt = 1:app.MAX_CAPTURE_RETRIES
                    try
                        img = snapshot(app.Cam);
                        captureSuccess = true;
                        break;
                    catch
                        if attempt < app.MAX_CAPTURE_RETRIES
                            pause(0.2);
                        end
                    end
                end
                
                % If all retries failed, attempt camera reconnection
                if ~captureSuccess
                    app.StatusLabel.Text = 'Camera timeout. Reconnecting...';
                    
                    reconnected = app.reconnectCamera();
                    if ~reconnected
                        pause(2.0);
                    end
                    
                    % Try one more capture after reconnection
                    try
                        img = snapshot(app.Cam);
                    catch
                        app.StatusLabel.Text = 'Capture failed after reconnection.';
                        pause(1.0);
                        img = imread('peppers.png');
                    end
                end
            else
                if app.InputVideoLoaded
                    while toc(app.LastImgFrameTime)<1/app.InputVideoReader.FrameRate
                        pause(0.001)
                    end
                    if not(app.InputVideoReader.hasFrame())
                       app.InputVideoReader.CurrentTime = 0;
                    end
                    img = readFrame(app.InputVideoReader);
                else
                    app.StatusLabel.Text = 'Video Not Loaded.';
                    pause(1.0);
                    img = imread('peppers.png');
                end
            end
            img = fliplr(img);
            app.LastImgFrameTime = tic();
        end

        % =========================
        % TOOLBAR CALLBACK
        % =========================
        function LoadNetToolClicked(app,~,~)
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


        function InputChoiceUpdated(app,~,~)                                              
            if strcmp(app.ImageInputSelectionDropdown.State,'off')
                [file, path] = uigetfile('*','Video Selection');
                if isequal(file, 0)
                    return;
                end
    
                fullPath = fullfile(path, file);
    
                app.InputVideoReader = VideoReader(fullPath);
                app.InputVideoLoaded = true;
            else
                try
                    app.Cam = webcam;
                    if isprop(app.Cam, 'Timeout')
                        app.Cam.Timeout = 10;
                    end
                    app.InputVideoLoaded = false;
                catch
                    uiconfirm(app.UIFigure, 'No webcam found. Connect one and retry.', 'Camera Error','Options',{'OK'});
                end
            end
            app.LastImgFrameTime = tic();
        end

        % =========================
        % SLIDER CALLBACKS
        % =========================
        function ConfThreshSliderChanged(app, ~)
            app.MinConfidence = app.ConfThreshSlider.Value / 100;
            app.ConfThreshLabel.Text = sprintf('Min Confidence: %d%%', round(app.ConfThreshSlider.Value));
            app.GapThreshSlider.Limits = [-100, 100];
            if app.GapThreshSlider.Value < -app.ConfThreshSlider.Value
                app.GapThreshSlider.Value = -app.ConfThreshSlider.Value;
            end
            app.GapThreshSlider.Limits = [-round(app.ConfThreshSlider.Value), 100-round(app.ConfThreshSlider.Value)];
            app.ConfGauge.ScaleColorLimits = [0, app.ConfThreshSlider.Value+app.GapThreshSlider.Value+0.0001; app.ConfThreshSlider.Value+app.GapThreshSlider.Value-0.0001, app.ConfThreshSlider.Value; app.ConfThreshSlider.Value-0.0001, 100];
        end

        function GapThreshSliderChanged(app, ~)
            if app.GapThreshSlider.Value > 0
                app.GapThreshSlider.Value = 0;
            end
            app.MinScoreDiff = abs(app.GapThreshSlider.Value) / 100;
            app.GapThreshLabel.Text = sprintf('Min Gap: %d%%', abs(round(app.GapThreshSlider.Value)));
            app.ConfGauge.ScaleColorLimits = [0, app.ConfThreshSlider.Value+app.GapThreshSlider.Value+0.0001; app.ConfThreshSlider.Value+app.GapThreshSlider.Value-0.0001, app.ConfThreshSlider.Value; app.ConfThreshSlider.Value-0.0001, 100];
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
            while app.IsRunning && isvalid(app.UIFigure)
                
                img = app.getImage();
                
                [h, w, ~] = size(img);

                boxSize = min([app.targetBoxSize, h, w]);
                x = round((w - boxSize)/2);
                if x < 1
                    x = 1;
                end
                y = round((h - boxSize)/2);
                if y < 1
                    y = 1;
                end
                rect = [x, y, boxSize-1, boxSize-1];

                % --- B. Classification (only every app.PREDICTION_INTERVAL seconds) ---
                if isempty(app.Net) || isempty(app.NetInputSize)
                    app.boxColor = 'white';
                    app.clearTop5Display()
                    app.StatusLabel.Text = 'Load a network from the toolbar to begin.';
                    app.ConfGauge.Value = 0;
                    app.CurrentCharLabel.Text = '-';

                else
                    timeSincePrediction = toc(app.LastPredictionTime);
    
                    if timeSincePrediction >= app.PREDICTION_INTERVAL
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
    
                        % Update top 5 display
                        classNames = app.Net.Layers(end).Classes;
                        app.updateTop5Display(scores, classNames);
                        
                        % Use cached values
                        maxScore = app.LastPredictedScore;
                        currentChar = app.LastPredictedChar;
                        scoreDiff = app.LastScoreDiff;
        
                        % Check if lock-in conditions are met
                        meetsThreshold = (maxScore >= app.MinConfidence) && (scoreDiff >= app.MinScoreDiff);
                        
                        isDifferentChar = ~strcmp(currentChar, app.LastLockedChar);
                        
                        canLockIn = meetsThreshold && isDifferentChar;
        
                        % --- C. Automatic lock-in ---    
                        if canLockIn && ~isempty(currentChar) && currentChar ~= '-'
                            % Automatically lock in the letter
                            app.appendToSentence(currentChar);
                            app.LastLockedChar = currentChar;
                            app.boxColor = 'green';
                            app.CurrentCharLabel.FontColor = 'green';
                            app.StatusLabel.Text = ['Locked: ' currentChar];
                        else
                            % Update status based on current state
                            if meetsThreshold && ~isDifferentChar
                                app.boxColor = 'cyan';
                                app.StatusLabel.Text = sprintf('Same as last (%s)', app.LastLockedChar);
                            elseif meetsThreshold
                                app.boxColor = 'cyan';
                                app.StatusLabel.Text = 'Ready...';
                                app.CurrentCharLabel.FontColor = '#e38902';
                            elseif maxScore < app.MinConfidence
                                app.LastLockedChar = '-';
                                app.boxColor = 'red';
                                app.CurrentCharLabel.FontColor = '#e38902';
                                app.StatusLabel.Text = sprintf('Conf: %.0f%% (need %.0f%%)', maxScore * 100, app.MinConfidence * 100);
                            else
                                app.boxColor = 'yellow';
                                app.CurrentCharLabel.FontColor = '#e38902';
                                app.StatusLabel.Text = sprintf('Gap: %.0f%% (need %.0f%%)', scoreDiff * 100, app.MinScoreDiff * 100);
                            end
                        end
    
                        app.CurrentCharLabel.Text = currentChar;
                        app.ConfGauge.Value = maxScore * 100;
                    end
                end

                % --- D. Update UI ---
                imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 6, 'Color', app.boxColor);
                imshow(imgDisplay, 'Parent', app.ImageAxes);
                
                drawnow limitrate;
            end

            % After While Loop Finishes - Delete App
            if ~isempty(app.Cam)
                delete(app.Cam);
                app.Cam = [];
            end
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
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

        function DeleteButtonPushed(app, ~)
            currentTxt = char(app.SentenceText);
            if ~isempty(currentTxt)
                app.SentenceText = string(currentTxt(1:end-1));
                app.SentenceLabel.Text = app.SentenceText;
            end
        end

        function UIFigureCloseRequest(app, ~)
            app.IsRunning = false;
        end
    end

    methods (Access = public)
        % =========================
        % CREATE COMPONENTS
        % =========================
        function createComponents(app)
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1100 650];
            app.UIFigure.Name = 'ASL Translator';
            app.UIFigure.Resize = 'off';
            app.UIFigure.Color = [0.92 0.93 0.94];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);
            app.UIFigure.KeyPressFcn = createCallbackFcn(app, @KeyPressed, true);

            % === TOOLBAR ===
            app.Toolbar = uitoolbar(app.UIFigure);

            icon = zeros(16,16,3);
            icon(:,:,2) = 0.6;
            icon(3:14,3:14,:) = 0.9;

            app.LoadNetTool = uipushtool(app.Toolbar);
            app.LoadNetTool.Tooltip= 'Load Network (.mat)';
            app.LoadNetTool.Icon = icon;
            app.LoadNetTool.ClickedCallback = @app.LoadNetToolClicked;

            app.ImageInputSelectionDropdown = uitoggletool(app.Toolbar);
            app.ImageInputSelectionDropdown.Icon = icon;
            app.ImageInputSelectionDropdown.Tooltip = 'on=webcam,off=video';
            app.ImageInputSelectionDropdown.ClickedCallback = @app.InputChoiceUpdated;
            app.ImageInputSelectionDropdown.State = 'on';

            % Define column geometry
            col1_x = 10;  col1_w = 340;
            col2_x = 360; col2_w = 490;
            col3_x = 860; col3_w = 230;
            base_y = 10; panel_h = 630;

            % === LEFT PANEL: Reference chart ===
            app.ChartPanel = uipanel(app.UIFigure);
            app.ChartPanel.Title = 'Reference Chart & Output';
            app.ChartPanel.Position = [col1_x base_y col1_w panel_h];

            imagePath = 'asl_chart.jpeg';
            if ~exist(imagePath, 'file')
                imwrite(zeros(300,300,3)+0.9, 'asl_chart_placeholder.jpeg');
                imagePath = 'asl_chart_placeholder.jpeg';
            end

            app.ChartImage = uiimage(app.ChartPanel);
            img_width = col1_w-10;
            img_height = img_width*487/630;
            app.ChartImage.Position = [5, panel_h-25-img_height, img_width, img_height];
            app.ChartImage.ImageSource = imagePath;
            app.ChartImage.ScaleMethod = 'fit';

            app.SentenceLabel = uilabel(app.ChartPanel);
            app.SentenceLabel.Position = [5, 60, img_width, panel_h-60-img_height-25-5];
            app.SentenceLabel.Text = '';                   
            app.SentenceLabel.BackgroundColor = 'white';
            app.SentenceLabel.FontSize = 24;
            app.SentenceLabel.FontWeight = 'bold';
            app.SentenceLabel.VerticalAlignment = 'bottom';
            app.SentenceLabel.WordWrap = 'on';

            button_width = (col1_w/2)-10;

            app.SpaceButton = uibutton(app.ChartPanel, 'push');
            app.SpaceButton.Position = [5, 5, button_width, 50];
            app.SpaceButton.Text = 'SPACE';
            app.SpaceButton.FontSize = 16;
            app.SpaceButton.ButtonPushedFcn = createCallbackFcn(app, @(src,event)app.appendToSentence(" "), true);

            app.DeleteButton = uibutton(app.ChartPanel, 'push');
            app.DeleteButton.Position = [button_width+15, 5, button_width, 50];
            app.DeleteButton.Text = 'BACKSPACE';
            app.DeleteButton.FontSize = 16;
            app.DeleteButton.ButtonPushedFcn = createCallbackFcn(app, @DeleteButtonPushed, true);

            % === MIDDLE PANEL: Camera ===
            app.CamPanel = uipanel(app.UIFigure);
            app.CamPanel.Title = 'Live Input';
            app.CamPanel.Position = [col2_x, base_y, col2_w, panel_h];

            app.ImageAxes = uiaxes(app.CamPanel);
            app.ImageAxes.Position = [5, 5, col2_w+9, panel_h-15];
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickMode = 'manual';
            app.ImageAxes.YTick = [];
            app.ImageAxes.YTickMode = 'manual';
            app.ImageAxes.DataAspectRatio = [1,1,1];
            app.ImageAxes.DataAspectRatioMode = 'manual';
            app.ImageAxes.Color = [0.95,0.95,0.95];
            colorbar(app.ImageAxes,'off');

            % === RIGHT PANEL: Results & controls ===
            app.ResultsPanel = uipanel(app.UIFigure);
            app.ResultsPanel.Title = 'Real-Time Analysis';
            app.ResultsPanel.Position = [col3_x, base_y, col3_w, panel_h];

            app.CurrentCharLabel = uilabel(app.ResultsPanel);
            app.CurrentCharLabel.Position = [10 490 200 100];
            app.CurrentCharLabel.Text = '-';
            app.CurrentCharLabel.FontSize = 90;
            app.CurrentCharLabel.FontWeight = 'bold';
            app.CurrentCharLabel.HorizontalAlignment = 'center';

            app.ConfGauge = uigauge(app.ResultsPanel, 'linear');
            app.ConfGauge.Position = [8, 435, col3_w-16, 40];
            app.ConfGauge.Limits = [0 100];
            app.ConfGauge.ScaleColors = [0.8 0 0; 1 0.8 0; 0 0.6 0];
            app.ConfGauge.ScaleColorLimits = [0 40; 40 70; 70 100];

            app.GaugeLabel = uilabel(app.ResultsPanel);
            app.GaugeLabel.Position = [5, 405, col3_w-10, 25];
            app.GaugeLabel.Text = 'Confidence %';
            app.GaugeLabel.HorizontalAlignment = 'center';
            app.GaugeLabel.VerticalAlignment = 'top';

            app.Top5Label = uilabel(app.ResultsPanel);
            app.Top5Label.Position = [10, 380, col3_w-20, 25];
            app.Top5Label.Text = 'Top 5 Predictions';
            app.Top5Label.FontWeight = 'bold';
            app.Top5Label.HorizontalAlignment = 'center';
            app.Top5Label.VerticalAlignment = 'bottom';

            app.Top5List = uilabel(app.ResultsPanel);
            app.Top5List.Position = [5, 255, col3_w-10, 120];
            app.Top5List.Text = '-';
            app.Top5List.FontSize = 14;
            app.Top5List.FontName = 'Consolas';
            app.Top5List.VerticalAlignment = 'top';
            app.Top5List.BackgroundColor = [0.95 0.95 0.95];

            % Confidence threshold slider
            app.ConfThreshLabel = uilabel(app.ResultsPanel);
            app.ConfThreshLabel.Position = [5, 225, col3_w-10, 25];
            app.ConfThreshLabel.Text = 'Min Confidence: 70%';
            app.ConfThreshLabel.HorizontalAlignment = 'center';
            app.ConfThreshLabel.VerticalAlignment = 'bottom';

            app.ConfThreshSlider = uislider(app.ResultsPanel);
            app.ConfThreshSlider.Position = [15, 215, col3_w-30, 3];
            app.ConfThreshSlider.Limits = [0 100];
            app.ConfThreshSlider.Value = 70;
            app.ConfThreshSlider.ValueChangedFcn = createCallbackFcn(app, @ConfThreshSliderChanged, true);

            % Gap threshold slider
            app.GapThreshLabel = uilabel(app.ResultsPanel);
            app.GapThreshLabel.Position = [5, 165, col3_w-10, 25];
            app.GapThreshLabel.Text = 'Min Gap: 40%';
            app.GapThreshLabel.HorizontalAlignment = 'center';
            app.GapThreshLabel.VerticalAlignment = 'bottom';

            app.GapThreshSlider = uislider(app.ResultsPanel);
            app.GapThreshSlider.Position = [15, 155, col3_w-30, 3];
            app.GapThreshSlider.Limits = [-70 30];
            app.GapThreshSlider.Value = -40;
            app.GapThreshSlider.ValueChangedFcn = createCallbackFcn(app, @GapThreshSliderChanged, true);

            app.StatusLabel = uilabel(app.ResultsPanel);
            app.StatusLabel.Position = [10, 5, col3_w-20, 100];
            app.StatusLabel.Text = ' ';
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.WordWrap = 'on';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.FontAngle = 'italic';
            app.StatusLabel.VerticalAlignment = 'top';

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