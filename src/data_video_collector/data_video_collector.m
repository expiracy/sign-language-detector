classdef data_video_collector < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        IDEdit              matlab.ui.control.NumericEditField
        IDEditFieldLabel    matlab.ui.control.Label
        StatusLabel         matlab.ui.control.Label
        PleasepresentLabel  matlab.ui.control.Label
        LetterLabel         matlab.ui.control.Label
        CaptureButton       matlab.ui.control.Button
        NameEdit            matlab.ui.control.EditField
        NameLabel           matlab.ui.control.Label
        ChartImage          matlab.ui.control.Image
        ImageAxes           matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        Cam             % Webcam object
        currentLetter   % Tracks letter to capture
        SpacePressed    % Boolean to detect a space press
        isRecording     % Detect if currently recording
        videoWriter

        % TIMING PROPERTIES
        recordTimer     % Stores the recording frame timer

        % TIMESTAMP TRACKING
        letterTimestamps        % Cell array: {Letter, StartFrame, EndFrame}
        currentLetterStartFrame % Start frame of current letter
        frameCount              % Current frame number

        % CONSTANTS
        FRAME_RATE = 30
        CROP_SIZE  = 450
        FIRST_LETTER = 'A'
        LAST_LETTER  = 'Z'
    
        COLOUR_OK = [0 1 0]
        COLOUR_WARN = [1 0 0]
        COLOUR_INFO = [0 0 1]
    end
    
    methods (Access = private)

        function rect = computeCenterRect(~, frame, targetBoxSize)
            [h, w, ~] = size(frame);
            boxSize = min([targetBoxSize, h, w]);
            x = max(1, round((w - boxSize)/2));
            y = max(1, round((h - boxSize)/2));
            rect = [x, y, boxSize-1, boxSize-1];
        end

        function img = drawOverlay(app, frame, rect)
            if app.SpacePressed
                color = 'yellow';
            else
                color = 'red';
            end

            img = insertShape(frame, 'Rectangle', rect, ...
                              'LineWidth', 4, 'Color', color);
        end
        
        function updatePreview(app, img)
            cla(app.ImageAxes);
            image(app.ImageAxes, fliplr(img));
            axis(app.ImageAxes, 'image');
            app.ImageAxes.XTick = [];
            app.ImageAxes.YTick = [];
        end
          
        % Loop to update camera feed on the Axes
        function updateCameraAndRecord(app)
            if ~isvalid(app.UIFigure)
                return;
            end
        
            try
                frame = snapshot(app.Cam);
        
                rect = computeCenterRect(app, frame, app.CROP_SIZE);
                img  = drawOverlay(app, frame, rect);
                updatePreview(app, img);
        
                app.frameCount = app.frameCount + 1;
        
                % Only write video when recording
                if app.isRecording
                    % Crop to the rectangle region
                    imgCropped = imcrop(fliplr(frame), rect);
                    imgResized = imresize(imgCropped, [224, 224]); 
                    writeVideo(app.videoWriter, imgResized);
                end
        
                drawnow limitrate;
        
            catch ME
                disp(getReport(ME));
            end
        end



        function UpdateStatus(app, msg, clr)
            arguments
                app  
                msg = ''
                clr = app.COLOUR_WARN
            end
            app.StatusLabel.Text = msg;
            app.StatusLabel.FontColor = clr;
            drawnow limitrate nocallbacks;
        end       


        function startVideoRecording(app)
            if isempty(app.Cam) || ~isvalid(app.Cam)
                UpdateStatus(app, 'Error: Camera not active.');
                return;
            end
            
            % Validate Name
            userName = strtrim(app.NameEdit.Value);
            if isempty(userName)
                UpdateStatus(app, 'Error: Please enter your Name.');
                return;
            end
            userName = regexprep(userName, '\s+', '_');
            
            % Create folder
            rootFolder = fullfile(pwd, ['data/test/video/', userName]);
            if ~exist(rootFolder, 'dir')
                mkdir(rootFolder);
            end
            
            % Generate filename
            fileNameStr = sprintf('ASL_AtoZ_%s_%s.mp4', userName, string(app.IDEdit.Value));
            fullFilePath = fullfile(rootFolder, fileNameStr);
            
            % Create video writer
            try
                app.videoWriter = VideoWriter(fullFilePath, 'MPEG-4');
                app.videoWriter.FrameRate = app.FRAME_RATE;
                open(app.videoWriter);
            catch ME
                UpdateStatus(app, ['Error: ' ME.message]);
                return;
            end
            
            % Initialize recording session
            app.frameCount = 0;
            app.currentLetterStartFrame = 0;
            app.letterTimestamps = {};
            app.currentLetter = app.FIRST_LETTER;
            app.LetterLabel.Text = app.FIRST_LETTER;
            app.isRecording = true;
            app.SpacePressed = false;
            
            % Change button text
            app.CaptureButton.Text = 'Stop Recording';
            
            UpdateStatus(app, 'Recording started! Hold SPACE to mark letter A', app.COLOUR_INFO);  % Blue
        end


        function stopVideoRecording(app)
            if ~app.isRecording
                return;
            end
            
            % Close video
            if ~isempty(app.videoWriter) && isvalid(app.videoWriter)
                close(app.videoWriter);
            end
            app.isRecording = false;
            
            % Save timestamps to CSV
            userName = strtrim(app.NameEdit.Value);
            userName = regexprep(userName, '\s+', '_');
            rootFolder = fullfile(pwd, ['data/test/video/', userName]);
            csvFileName = sprintf('ASL_Timestamps_%s_%s.csv', userName, string(app.IDEdit.Value));
            csvFilePath = fullfile(rootFolder, csvFileName);
            
            % Write timestamps
            try
                fid = fopen(csvFilePath, 'w');
                fprintf(fid, 'Letter,StartFrame,EndFrame,TotalFrames\n');
                for i = 1:length(app.letterTimestamps)
                    letter = app.letterTimestamps{i}{1};
                    startFrame = app.letterTimestamps{i}{2};
                    endFrame = app.letterTimestamps{i}{3};
                    fprintf(fid, '%s,%d,%d,%d\n', letter, startFrame, endFrame, endFrame - startFrame);
                end
                fclose(fid);
                
                UpdateStatus(app, sprintf('Complete! %d frames recorded, %d letters marked.', ...
                    app.frameCount, length(app.letterTimestamps)), app.COLOUR_OK);
            catch ME
                UpdateStatus(app, 'Video saved but CSV export failed.', [1 0.5 0]);
            end
            
            % Reset UI
            app.CaptureButton.Text = 'Start Recording';
            app.currentLetter = app.FIRST_LETTER;
            app.LetterLabel.Text = app.FIRST_LETTER;
            app.IDEdit.Value = app.IDEdit.Value + 1;
        end

    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.isRecording = false;
            app.SpacePressed = false;
            app.currentLetter = 'A';
            app.frameCount = 0;
            app.letterTimestamps = {};
            app.currentLetterStartFrame = 0;
            
            % 1. Setup Camera
            try
                app.Cam = webcam;
            catch
                uialert(app.UIFigure, 'No Webcam found!', 'Error');
                app.CaptureButton.Enable = 'off';
                return;
            end
            
            % 3. Start the live preview loop
            app.recordTimer = timer('ExecutionMode', 'fixedRate', ...
                                   'Period', 1/app.FRAME_RATE, ... % 30 fps
                                   'TimerFcn', @(~,~)updateCameraAndRecord(app));
            start(app.recordTimer);
        end

        % Window key press function: UIFigure
        function UIFigureWindowKeyPress(app, event)
            if strcmp(event.Key, 'space') && app.isRecording && ~app.SpacePressed
                app.SpacePressed = true;
                
                % Calculate timestamp based on frames written
                app.currentLetterStartFrame = app.frameCount;
                
                UpdateStatus(app, sprintf('Marking %s...', app.currentLetter), app.COLOUR_WARN);  % Red while held
            end
        end

        % Window key release function: UIFigure
        function UIFigureWindowKeyRelease(app, event)
            if strcmp(event.Key, 'space') && app.isRecording && app.SpacePressed
                app.SpacePressed = false;
                
                % Save {Letter, StartFrame, EndFrame}
                app.letterTimestamps{end+1} = {app.currentLetter, app.currentLetterStartFrame, app.frameCount};
                
                % Move to next letter
                if app.currentLetter < app.LAST_LETTER
                    app.currentLetter = char(app.currentLetter + 1);
                    app.LetterLabel.Text = app.currentLetter;
                    UpdateStatus(app, sprintf('%s saved! (Frames %d-%d) Hold SPACE for %s', ...
                char(app.currentLetter - 1), app.currentLetterStartFrame, app.frameCount, app.currentLetter), app.COLOUR_OK);  % Green
                else
                    % Reached Z - automatically stop recording and save
                    UpdateStatus(app, 'Z saved! Finishing recording...', app.COLOUR_OK);
                    stopVideoRecording(app);
                end
            end
        end

        % Close request function: UIFigure
        function UIFigureCloseRequest(app, event)
            if ~isempty(app.recordTimer) && isvalid(app.recordTimer)
                stop(app.recordTimer);
                delete(app.recordTimer);
            end
            if ~isempty(app.Cam)
                clear app.Cam;
            end
            delete(app);
        end

        % Button pushed function: CaptureButton
        function CaptureButtonPushed(app, event)
            if ~app.isRecording
                % Start continuous video recording
                startVideoRecording(app);
            else
                % Stop video recording and save
                stopVideoRecording(app);
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 772 480];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);
            app.UIFigure.WindowKeyPressFcn = createCallbackFcn(app, @UIFigureWindowKeyPress, true);
            app.UIFigure.WindowKeyReleaseFcn = createCallbackFcn(app, @UIFigureWindowKeyRelease, true);

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.UIFigure);
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickLabel = '';
            app.ImageAxes.YTick = [];
            app.ImageAxes.BoxStyle = 'full';
            app.ImageAxes.LineWidth = 0.1;
            app.ImageAxes.Position = [447 140 300 306];

            % Create ChartImage
            app.ChartImage = uiimage(app.UIFigure);
            app.ChartImage.Position = [28 192 364 268];
            app.ChartImage.ImageSource = fullfile(pathToMLAPP, 'asl_chart.jpeg');

            % Create NameLabel
            app.NameLabel = uilabel(app.UIFigure);
            app.NameLabel.HorizontalAlignment = 'right';
            app.NameLabel.Position = [200 119 40 22];
            app.NameLabel.Text = 'Name:';

            % Create NameEdit
            app.NameEdit = uieditfield(app.UIFigure, 'text');
            app.NameEdit.Placeholder = 'Enter Name';
            app.NameEdit.Position = [255 119 100 22];

            % Create CaptureButton
            app.CaptureButton = uibutton(app.UIFigure, 'push');
            app.CaptureButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureButtonPushed, true);
            app.CaptureButton.Position = [468 50 273 60];
            app.CaptureButton.Text = 'Start Recording';

            % Create LetterLabel
            app.LetterLabel = uilabel(app.UIFigure);
            app.LetterLabel.HorizontalAlignment = 'center';
            app.LetterLabel.FontSize = 48;
            app.LetterLabel.FontWeight = 'bold';
            app.LetterLabel.FontColor = [1 0 0];
            app.LetterLabel.Position = [66 47 58 63];
            app.LetterLabel.Text = 'A';

            % Create PleasepresentLabel
            app.PleasepresentLabel = uilabel(app.UIFigure);
            app.PleasepresentLabel.HorizontalAlignment = 'center';
            app.PleasepresentLabel.FontSize = 14;
            app.PleasepresentLabel.Position = [44 119 102 22];
            app.PleasepresentLabel.Text = 'Please present:';

            % Create StatusLabel
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.Position = [11 11 497 22];
            app.StatusLabel.Text = '';

            % Create IDEditFieldLabel
            app.IDEditFieldLabel = uilabel(app.UIFigure);
            app.IDEditFieldLabel.HorizontalAlignment = 'right';
            app.IDEditFieldLabel.Position = [215 88 25 22];
            app.IDEditFieldLabel.Text = 'ID';

            % Create IDEdit
            app.IDEdit = uieditfield(app.UIFigure, 'numeric');
            app.IDEdit.Limits = [1 Inf];
            app.IDEdit.RoundFractionalValues = 'on';
            app.IDEdit.ValueDisplayFormat = '%.0f';
            app.IDEdit.HorizontalAlignment = 'left';
            app.IDEdit.Position = [255 88 100 22];
            app.IDEdit.Value = 1;

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = data_video_collector

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