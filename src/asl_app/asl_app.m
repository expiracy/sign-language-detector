classdef asl_app < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure        matlab.ui.Figure

        % --- Left Panel (Chart) ---
        ChartPanel      matlab.ui.container.Panel
        ChartImage      matlab.ui.control.Image
        SentenceLabel   matlab.ui.control.Label
        SpaceButton     matlab.ui.control.Button
        DeleteButton    matlab.ui.control.Button
        CopyButton    matlab.ui.control.Button

        % --- Middle Panel (Camera & Sentence) ---
        CamPanel        matlab.ui.container.Panel
        ImageInputSelectionDropdown matlab.ui.control.DropDown
        ImageInputSelectionToggleButton matlab.ui.control.StateButton
        ImageAxes       matlab.ui.control.UIAxes

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
        NewNetSelectionButton matlab.ui.control.Button
        NetSelectionDropdown matlab.ui.control.DropDown
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
        LoadedVideoFiles
        VideoPlaceholderImg

        % Configurable thresholds
        MinConfidence       double = 0.7
        MinScoreDiff        double = 0.4
        targetBoxSize       uint64 = 450
        PREDICTION_INTERVAL double = 1.0
        MAX_CAPTURE_RETRIES uint64 = 3
    end

    methods (Access = private)

        % =========================
        % STARTUP
        % =========================
        function startupFcn(app)
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

            % Start camera with timeout configuration
            app.InputChoiceToggleUpdated()

            app.IsRunning = true;

            app.InputVideoLoaded = false;
            app.LoadedVideoFiles = {'Load New...'};
            app.LastImgFrameTime = tic();

            imagePath = 'PlaceHolderImg.png';
            if ~exist(imagePath, 'file')
                imwrite(zeros(300,300,3)+0.9, 'PlaceHolderImg_placeholder.png');
                imagePath = 'PlaceHolderImg_placeholder.png';
            end
            app.VideoPlaceholderImg = fliplr(imread(imagePath));

            % Start loop
            drawnow
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

            if ~app.ImageInputSelectionToggleButton.Value
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
                img = fliplr(img);
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
                    img = app.VideoPlaceholderImg;
                end
            end

            app.LastImgFrameTime = tic();
        end

        % =========================
        % INPUT SELECTION CALLBACKS
        % =========================
        function LoadNetButtonClicked(app)
            [file, path] = uigetfile({'*.mat','MAT-files (*.mat)'}, 'Select a trained network');
            if isequal(file, 0)
                return;
            end

            fullPath = fullfile(path, file);
            app.NetSelectionDropdown.Items = [app.NetSelectionDropdown.Items,{fullPath}];
            app.NetSelectionDropdown.Value = fullPath;
            app.NetSelectionUpdated();
        end

        function NetSelectionUpdated(app)
            fullPath = app.NetSelectionDropdown.Value;

            if ~strcmp(fullPath,'No Net Loaded')
                try
                    data = load(fullPath);
                    [net, ~] = app.findNetworkInLoadedStruct(data);
    
                    if ~(isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork'))
                        error('Unsupported network type. Save a SeriesNetwork or DAGNetwork.');
                    end
    
                    app.Net = net;
                    app.NetInputSize = app.getNetInputSize(net);
    
                    % Reset state when loading a new network
                    app.LastLockedChar = '';
                    app.LastPredictionTime = tic;
    
    
                catch err
                    uialert(app.UIFigure, err.message, 'Network Load Error');
                end
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

        
        function InputChoiceToggleUpdated(app)                                              
            if app.ImageInputSelectionToggleButton.Value
                app.ImageInputSelectionToggleButton.Text = 'Video';
                app.ImageInputSelectionDropdown.Items = app.LoadedVideoFiles;
            else
                app.ImageInputSelectionToggleButton.Text = 'Webcam';
                app.InputVideoLoaded = false;
                try
                    app.ImageInputSelectionDropdown.Items = webcamlist();
                catch
                    uiconfirm(app.UIFigure, 'No webcam found. Connect one and retry.', 'Camera Error','Options',{'OK'});
                    app.ImageInputSelectionToggleButton.Value = True;
                    app.InputChoiceToggleUpdated();
                end
            end
            app.InputChoiceDropdownUpdated();
        end

        function InputChoiceDropdownUpdated(app)                                              
            if app.ImageInputSelectionToggleButton.Value
                if strcmp(app.ImageInputSelectionDropdown.Value,'Load New...')
                    [file, path] = uigetfile('*','Video Selection');
                    if isequal(file, 0)
                        return;
                    end
                    fullPath = fullfile(path, file);
                    app.LoadedVideoFiles = [{fullPath}; app.LoadedVideoFiles];
                    app.ImageInputSelectionDropdown.Items = app.LoadedVideoFiles;
                    app.ImageInputSelectionDropdown.Value = fullPath;
                end

                app.InputVideoReader = VideoReader(app.ImageInputSelectionDropdown.Value);
                app.InputVideoLoaded = true;
            else
                app.Cam = webcam(app.ImageInputSelectionDropdown.ValueIndex);
                if isprop(app.Cam, 'Timeout')
                    app.Cam.Timeout = 10;
                end
            end
            app.LastImgFrameTime = tic();
        end

        % =========================
        % SLIDER CALLBACKS
        % =========================
        function ConfThreshSliderChanged(app)
            app.MinConfidence = app.ConfThreshSlider.Value / 100;
            app.ConfThreshLabel.Text = sprintf('Min Confidence: %d%%', round(app.ConfThreshSlider.Value));
            app.GapThreshSlider.Limits = [-100, 100];
            if app.GapThreshSlider.Value < -app.ConfThreshSlider.Value
                app.GapThreshSlider.Value = -app.ConfThreshSlider.Value;
            end
            app.GapThreshSlider.Limits = [-round(app.ConfThreshSlider.Value), 100-round(app.ConfThreshSlider.Value)];
            app.ConfGauge.ScaleColorLimits = [0, app.ConfThreshSlider.Value+app.GapThreshSlider.Value+0.0001; app.ConfThreshSlider.Value+app.GapThreshSlider.Value-0.0001, app.ConfThreshSlider.Value; app.ConfThreshSlider.Value-0.0001, 100];
        end

        function GapThreshSliderChanged(app)
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
                    app.StatusLabel.Text = 'Load a network to begin.';
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
                            app.CurrentCharLabel.Text = currentChar;
                        else
                            % Update status based on current state
                            if meetsThreshold && ~isDifferentChar
                                app.boxColor = 'cyan';
                                app.CurrentCharLabel.FontColor = 'cyan';
                                app.StatusLabel.Text = sprintf('Same as last (%s)', app.LastLockedChar);
                                app.CurrentCharLabel.Text = currentChar;
                            elseif meetsThreshold
                                app.boxColor = 'cyan';
                                app.StatusLabel.Text = 'Ready...';
                                app.CurrentCharLabel.FontColor = 'cyan';
                                app.CurrentCharLabel.Text = currentChar;
                            elseif maxScore < (app.MinConfidence-app.MinScoreDiff)
                                app.LastLockedChar = '-';
                                app.boxColor = 'red';
                                app.CurrentCharLabel.FontColor =  'red';
                                app.StatusLabel.Text = sprintf('Conf: %.0f%% (need %.0f%%)', maxScore * 100, app.MinConfidence * 100);
                                app.CurrentCharLabel.Text = '-';
                            else
                                app.boxColor = 'yellow';
                                app.CurrentCharLabel.FontColor = 'yellow';
                                app.StatusLabel.Text = sprintf('Gap: %.0f%% (need %.0f%%)', scoreDiff * 100, app.MinScoreDiff * 100);
                                app.CurrentCharLabel.Text = currentChar;
                            end
                        end
  
                        app.ConfGauge.Value = maxScore * 100;
                    end
                end

                % --- D. Update UI ---
                delete(app.ImageAxes.Children);
                imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 6, 'Color', app.boxColor);
                imgSize = size(imgDisplay,[2,1]);
                axisSize = app.ImageAxes.InnerPosition([3,4]);
                outputImgSize = axisSize*min(imgSize./axisSize);
                imgDisplay = imcrop(imgDisplay, [(imgSize-outputImgSize)/2,outputImgSize]);
                image('XData',[0,axisSize(1)],'YData',[axisSize(2),0],'CData',imgDisplay,'Parent', app.ImageAxes);
                
                drawnow();
                
            end
                
            % After While Loop Finishes - Delete App
            if ~isempty(app.Cam)
                delete(app.Cam);
                app.Cam = [];
            end

            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end

            delete(app);
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

        function DeleteButtonPushed(app)
            currentTxt = char(app.SentenceText);
            if ~isempty(currentTxt)
                app.SentenceText = string(currentTxt(1:end-1));
                app.SentenceLabel.Text = app.SentenceText;
            end
        end

        function CopyButtonPushed(app)
            clipboard('copy',app.SentenceLabel.Text);
        end

        function UIFigureCloseRequest(app)
            app.IsRunning = false;
        end

        % =========================
        % UI RESIZE
        % =========================
        function UIResize(app,~,~)
            pause(0.1);
            size=app.UIFigure.InnerPosition([3,4]);

            % Define column geometry
            col1_2_w_ratio = (size(1)-250)/(340+490);
            col1_x = 5;  
            col1_w = col1_2_w_ratio*340;
            col2_x = col1_w+col1_x+5; 
            col2_w = col1_2_w_ratio*490;
            col3_x = size(1)-230-5; 
            col3_w = 230;
            base_y = 5;
            panel_h = size(2)-10;

            % === LEFT PANEL: Reference chart & Output ===
            app.ChartPanel.Position = [col1_x base_y col1_w panel_h];
            img_width = col1_w-10;
            img_height = img_width*487/630;
            app.ChartImage.Position = [5, panel_h-25-img_height, img_width, img_height];
            app.SentenceLabel.Position = [5, 60, img_width, panel_h-60-img_height-25-5];

            button_width = ((col1_w-20)/3);
            app.SpaceButton.Position = [5, 5, button_width, 50];
            app.DeleteButton.Position = [button_width+10, 5, button_width, 50];
            app.CopyButton.Position = [button_width*2+15, 5, button_width, 50];

            % === MIDDLE PANEL: Camera ===
            app.CamPanel.Position = [col2_x, base_y, col2_w, panel_h];
            app.ImageInputSelectionToggleButton.Position = [5, panel_h-45, 100, 20];
            app.ImageInputSelectionDropdown.Position = [110, panel_h-45, col2_w-115, 20];

            delete(app.ImageAxes);
           
            app.ImageAxes = uiaxes(app.CamPanel);
            app.ImageAxes.Units = 'pixels';
            app.ImageAxes.PositionConstraint = 'innerposition';
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickMode = 'manual';
            app.ImageAxes.YTick = [];
            app.ImageAxes.YTickMode = 'manual';
            app.ImageAxes.DataAspectRatioMode = 'auto';
            app.ImageAxes.PlotBoxAspectRatioMode = 'auto';
            app.ImageAxes.XLimMode = 'auto';
            app.ImageAxes.YLimMode = 'auto';
            app.ImageAxes.XLimitMethod = 'tight';
            app.ImageAxes.YLimitMethod = 'tight';
            app.ImageAxes.Color = [0.95,0.95,0.95];
            colorbar(app.ImageAxes,'off');
            app.ImageAxes.InnerPosition = [5, 5, col2_w-10, panel_h-55];

            % === RIGHT PANEL: Results & controls ===
            app.ResultsPanel.Position = [col3_x, base_y, col3_w, panel_h];
            app.CurrentCharLabel.Position = [10 490 200 100];
            app.ConfGauge.Position = [8, 435, col3_w-16, 40];
            app.GaugeLabel.Position = [5, 405, col3_w-10, 25];
            app.Top5Label.Position = [10, 380, col3_w-20, 25];
            app.Top5List.Position = [5, 255, col3_w-10, 120];
            app.ConfThreshLabel.Position = [5, 225, col3_w-10, 25];
            app.ConfThreshSlider.Position = [15, 215, col3_w-30, 3];
            app.GapThreshLabel.Position = [5, 165, col3_w-10, 25];
            app.GapThreshSlider.Position = [15, 155, col3_w-30, 3];
            app.StatusLabel.Position = [10, 55, col3_w-20, 55];
            app.NewNetSelectionButton.Position = [5, 30, col3_w-10, 20];
            app.NetSelectionDropdown.Position = [5, 5, col3_w-10, 20];
        end
    end

    methods (Access = public)
        % =========================
        % CREATE COMPONENTS
        % =========================
        function createComponents(app)
            % === FIGURE ===
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [0, 0, 1280, 720];
            app.UIFigure.Name = 'ASL Translator';
            app.UIFigure.Resize = 'on';
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Color = [0.92 0.93 0.94];
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, false);
            app.UIFigure.KeyPressFcn = createCallbackFcn(app, @KeyPressed, true);
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @UIResize, false);

            % === LEFT PANEL: Reference chart & Output ===
            app.ChartPanel = uipanel(app.UIFigure);
            app.ChartPanel.Title = 'Reference Chart & Output';

            imagePath = 'asl_chart.jpeg';
            if ~exist(imagePath, 'file')
                imwrite(zeros(300,300,3)+0.9, 'asl_chart_placeholder.jpeg');
                imagePath = 'asl_chart_placeholder.jpeg';
            end

            app.ChartImage = uiimage(app.ChartPanel);
            app.ChartImage.ImageSource = imagePath;
            app.ChartImage.ScaleMethod = 'fit';

            app.SentenceLabel = uilabel(app.ChartPanel);
            app.SentenceLabel.Text = '';                   
            app.SentenceLabel.BackgroundColor = 'white';
            app.SentenceLabel.FontSize = 24;
            app.SentenceLabel.FontWeight = 'bold';
            app.SentenceLabel.VerticalAlignment = 'bottom';
            app.SentenceLabel.WordWrap = 'on';

            app.SpaceButton = uibutton(app.ChartPanel, 'push');
            app.SpaceButton.Text = 'SPACE';
            app.SpaceButton.FontSize = 16;
            app.SpaceButton.ButtonPushedFcn = createCallbackFcn(app, @(~,~)app.appendToSentence(" "), false);

            app.DeleteButton = uibutton(app.ChartPanel, 'push');
            app.DeleteButton.Text = 'BACKSPACE';
            app.DeleteButton.FontSize = 16;
            app.DeleteButton.ButtonPushedFcn = createCallbackFcn(app, @DeleteButtonPushed, false);

            app.CopyButton = uibutton(app.ChartPanel, 'push');
            app.CopyButton.Text = 'COPY';
            app.CopyButton.FontSize = 16;
            app.CopyButton.ButtonPushedFcn = createCallbackFcn(app, @CopyButtonPushed, false);

            % === MIDDLE PANEL: Camera ===
            app.CamPanel = uipanel(app.UIFigure);
            app.CamPanel.Title = 'Live Input';

            app.ImageAxes = uiaxes(app.CamPanel);

            app.ImageInputSelectionToggleButton = uibutton(app.CamPanel, 'state');
            app.ImageInputSelectionToggleButton.ValueChangedFcn = createCallbackFcn(app, @InputChoiceToggleUpdated, false);
            app.ImageInputSelectionToggleButton.Text = 'Webcam';
            app.ImageInputSelectionToggleButton.Value = false;

            app.ImageInputSelectionDropdown = uidropdown(app.CamPanel);
            app.ImageInputSelectionDropdown.Items = {'No Cameras Found'};
            app.ImageInputSelectionDropdown.ValueChangedFcn = createCallbackFcn(app, @InputChoiceDropdownUpdated, false);
            app.ImageInputSelectionDropdown.DropDownOpeningFcn = createCallbackFcn(app, @InputChoiceDropdownUpdated, false);
            app.ImageInputSelectionDropdown.Value = 'No Cameras Found';

            % === RIGHT PANEL: Results & controls ===
            app.ResultsPanel = uipanel(app.UIFigure);
            app.ResultsPanel.Title = 'Real-Time Analysis';

            app.CurrentCharLabel = uilabel(app.ResultsPanel);
            app.CurrentCharLabel.Text = '-';
            app.CurrentCharLabel.FontSize = 90;
            app.CurrentCharLabel.FontWeight = 'bold';
            app.CurrentCharLabel.HorizontalAlignment = 'center';

            app.ConfGauge = uigauge(app.ResultsPanel, 'linear');
            app.ConfGauge.Limits = [0 100];
            app.ConfGauge.ScaleColors = [0.8 0 0; 1 0.8 0; 0 0.6 0];
            app.ConfGauge.ScaleColorLimits = [0 30; 30 70; 70 100];

            app.GaugeLabel = uilabel(app.ResultsPanel);
            app.GaugeLabel.Text = 'Confidence %';
            app.GaugeLabel.HorizontalAlignment = 'center';
            app.GaugeLabel.VerticalAlignment = 'top';

            app.Top5Label = uilabel(app.ResultsPanel);
            app.Top5Label.Text = 'Top 5 Predictions';
            app.Top5Label.FontWeight = 'bold';
            app.Top5Label.HorizontalAlignment = 'center';
            app.Top5Label.VerticalAlignment = 'bottom';

            app.Top5List = uilabel(app.ResultsPanel);
            app.Top5List.Text = '-';
            app.Top5List.FontSize = 14;
            app.Top5List.FontName = 'Consolas';
            app.Top5List.VerticalAlignment = 'top';
            app.Top5List.BackgroundColor = [0.95 0.95 0.95];

            % Confidence threshold slider
            app.ConfThreshLabel = uilabel(app.ResultsPanel);
            app.ConfThreshLabel.Text = 'Min Confidence: 70%';
            app.ConfThreshLabel.HorizontalAlignment = 'center';
            app.ConfThreshLabel.VerticalAlignment = 'bottom';

            app.ConfThreshSlider = uislider(app.ResultsPanel);
            app.ConfThreshSlider.Limits = [0 100];
            app.ConfThreshSlider.Value = 70;
            app.ConfThreshSlider.ValueChangedFcn = createCallbackFcn(app, @ConfThreshSliderChanged, false);

            % Gap threshold slider
            app.GapThreshLabel = uilabel(app.ResultsPanel);
            app.GapThreshLabel.Text = 'Min Gap: 40%';
            app.GapThreshLabel.HorizontalAlignment = 'center';
            app.GapThreshLabel.VerticalAlignment = 'bottom';

            app.GapThreshSlider = uislider(app.ResultsPanel);
            app.GapThreshSlider.Limits = [-70 30];
            app.GapThreshSlider.Value = -40;
            app.GapThreshSlider.ValueChangedFcn = createCallbackFcn(app, @GapThreshSliderChanged, false);

            app.StatusLabel = uilabel(app.ResultsPanel);
            app.StatusLabel.Text = ' ';
            app.StatusLabel.HorizontalAlignment = 'center';
            app.StatusLabel.WordWrap = 'on';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.FontAngle = 'italic';
            app.StatusLabel.VerticalAlignment = 'top';

            app.NewNetSelectionButton = uibutton(app.ResultsPanel);
            app.NewNetSelectionButton.ButtonPushedFcn = createCallbackFcn(app, @LoadNetButtonClicked, false);
            app.NewNetSelectionButton.Text = 'Load New Net';

            app.NetSelectionDropdown = uidropdown(app.ResultsPanel);
            app.NetSelectionDropdown.Items = {'No Net Loaded'};
            app.NetSelectionDropdown.ValueChangedFcn = createCallbackFcn(app, @NetSelectionUpdated, false);
            app.NetSelectionDropdown.Value = 'No Net Loaded';
    
            %app.UIResize();
            app.UIFigure.Visible = 'on';
        end
    end

    methods (Access = public)
        % =========================
        % CONSTRUCTOR
        % =========================
        function app = asl_app
            createComponents(app);
            registerApp(app, app.UIFigure);
            runStartupFcn(app, @startupFcn);
        end
    end
end