classdef data_collector < matlab.apps.AppBase

    properties (Access = public)
        UIFigure      matlab.ui.Figure
        ChartImage    matlab.ui.control.Image 
        ImageAxes     matlab.ui.control.UIAxes 
        
        % Controls
        NameLabel     matlab.ui.control.Label
        NameEdit      matlab.ui.control.EditField
        CharLabel     matlab.ui.control.Label
        CharEdit      matlab.ui.control.EditField
        IDLabel       matlab.ui.control.Label      
        IDEdit        matlab.ui.control.NumericEditField 
        CaptureButton matlab.ui.control.Button 
        
        % Status & Info
        CountLabel    matlab.ui.control.Label
        StatusLabel   matlab.ui.control.Label
        Instruction   matlab.ui.control.Label
    end

    properties (Access = private)
        Cam
        IsCapturing
        CurrentCount
        
        % Timing Properties
        SessionTimer 
        LastCaptureTime
    end

    methods (Access = private)

        function startupFcn(app)
            % 1. Setup Camera
            try
                app.Cam = webcam;
            catch
                uialert(app.UIFigure, 'No Webcam found!', 'Error');
                app.CaptureButton.Enable = 'off';
                return;
            end
            
            % 2. Initialise Timer for Debouncing (0.25s delay)
            app.SessionTimer = tic; 
            app.LastCaptureTime = -1;
            
            % 3. Start the live preview loop
            app.IsCapturing = true;
            app.updatePreviewLoop(); 
        end

        % Loop to update camera feed on the Axes
        function updatePreviewLoop(app)
            targetBoxSize = 450;
            
            while app.IsCapturing && isvalid(app.UIFigure)
                try
                    img = snapshot(app.Cam);
                    img = fliplr(img); % Mirror view
                    [h, w, ~] = size(img);
                    
                    % Calculate centre crop rectangle
                    boxSize = min([targetBoxSize, h, w]);
                    x = round((w - boxSize)/2); if x < 1, x = 1; end
                    y = round((h - boxSize)/2); if y < 1, y = 1; end
                    rect = [x, y, boxSize-1, boxSize-1];
                    
                    % Hand placement area
                    imgDisplay = insertShape(img, 'Rectangle', rect, 'LineWidth', 4, 'Color', 'yellow');
                    
                    image(app.ImageAxes, imgDisplay);
                    app.ImageAxes.XTick = [];
                    app.ImageAxes.YTick = [];
                    app.ImageAxes.Box = 'on'; 
                    drawnow limitrate;
                catch
                    app.IsCapturing = false;
                end
            end
        end

        % Core Function to Save Image
        function captureImage(app)
            if isempty(app.Cam) || ~isvalid(app.Cam)
                 app.StatusLabel.Text = 'Error: Camera not active.';
                 return;
            end
            
            % Debounce Check
            currentTime = toc(app.SessionTimer);
            if (currentTime - app.LastCaptureTime) < 0.25
                return; 
            end
            % Update the last capture time
            app.LastCaptureTime = currentTime;

            % Validation
            
            % 1. Validate Name
            userName = strtrim(app.NameEdit.Value);
            if isempty(userName)
                app.StatusLabel.Text = 'Error: Please enter your Name.';
                app.StatusLabel.FontColor = [1 0 0];
                return;
            end
            userName = regexprep(userName, '\s+', '_');

            % 2. Validate Character
            targetChar = upper(strtrim(app.CharEdit.Value));
            if isempty(targetChar) || length(targetChar) > 1 || targetChar < 'A' || targetChar > 'Z'
                app.StatusLabel.Text = 'Error: Please enter a single letter (A-Z).';
                app.StatusLabel.FontColor = [1 0 0];
                return;
            end
            
            % 3. Get Current ID
            currentID = app.IDEdit.Value;
            
            % Folder Saving
            
            % Root: ASL_DATA_NAME / LETTER
            rootFolder = fullfile(pwd, ['data/datasets_v2/ASL_DATA_', userName]);
            savePath = fullfile(rootFolder, targetChar);
            
            if ~exist(savePath, 'dir')
                mkdir(savePath);
            end
            
            % Capture and Save
            
            % Generate Filename: Character_Name_ID.jpg
            fileNameStr = sprintf('%s_%s_%d.jpg', targetChar, userName, currentID);
            fullFilePath = fullfile(savePath, fileNameStr);
            
            if exist(fullFilePath, 'file')
                app.StatusLabel.Text = ['Warning: ID ' num2str(currentID) ' already exists! Overwriting...'];
                app.StatusLabel.FontColor = [1 0.5 0];
                pause(0.1);
            end

            % Process Image - captures the region inside the rectangle
            img = snapshot(app.Cam);
            imgFlipped = fliplr(img);
            [h, w, ~] = size(imgFlipped);
            
            % Calculate centre crop rectangle
            targetBoxSize = 450;
            boxSize = min([targetBoxSize, h, w]);
            x = round((w - boxSize)/2); if x < 1, x = 1; end
            y = round((h - boxSize)/2); if y < 1, y = 1; end
            rect = [x, y, boxSize-1, boxSize-1];
            
            % Crop to the rectangle region
            imgCropped = imcrop(imgFlipped, rect);
            imgResized = imresize(imgCropped, [224, 224]); 
            imwrite(imgResized, fullFilePath);
            
            % UI Feedback and Updates
            
            % Flash Green
            app.ImageAxes.XColor = 'green';
            app.ImageAxes.YColor = 'green';
            app.ImageAxes.LineWidth = 3;
            
            % Increment Global Session Count
            app.CurrentCount = app.CurrentCount + 1;
            app.CountLabel.Text = "Session Count: " + app.CurrentCount;
            
            % Increment ID for next photo
            app.IDEdit.Value = currentID + 1;
            
            % Update Status
            app.StatusLabel.Text = "Saved: " + fileNameStr;
            app.StatusLabel.FontColor = [0 0.5 0]; % Green
            
            % Reset Flash
            pause(0.1);
            app.ImageAxes.XColor = [0.15 0.15 0.15]; 
            app.ImageAxes.YColor = [0.15 0.15 0.15];
            app.ImageAxes.LineWidth = 0.5;
        end

        % Callbacks

        function UIFigureWindowKeyPress(app, event)
            if strcmp(event.Key, 'space')
                app.captureImage();
            end
        end
        
        function CaptureButtonPushed(app, event)
             app.captureImage();
        end

        function UIFigureCloseRequest(app, event)
            app.IsCapturing = false; 
            pause(0.1); 
            if ~isempty(app.Cam) && isvalid(app.Cam)
                delete(app.Cam); 
            end
            delete(app.UIFigure); 
        end
    end

    % App Layout Initialisation
    methods (Access = public)
        function createComponents(app)
            % Create UIFigure
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 900 550]; 
            app.UIFigure.Name = 'ASL Data Collector';
            app.UIFigure.Color = [0.94 0.94 0.94];
            app.UIFigure.WindowKeyPressFcn = createCallbackFcn(app, @UIFigureWindowKeyPress, true);
            app.UIFigure.CloseRequestFcn = createCallbackFcn(app, @UIFigureCloseRequest, true);

            %Left Panel: ASL Chart
            imagePath = 'asl_chart.jpeg'; 
            if ~exist(imagePath, 'file')
                 dummyImg = zeros(400,400,3) + 0.9;
                 imwrite(dummyImg, 'asl_chart_placeholder.jpeg');
                 imagePath = 'asl_chart_placeholder.jpeg';
                 warning('asl_chart.jpeg not found. Using placeholder.');
            end
            
            app.ChartImage = uiimage(app.UIFigure);
            app.ChartImage.Position = [20 80 450 450];
            app.ChartImage.ImageSource = imagePath;
            app.ChartImage.ScaleMethod = 'fit';

            % Right Panel: Camera & Controls
            
            app.ImageAxes = uiaxes(app.UIFigure);
            app.ImageAxes.Position = [500 230 350 300];
            app.ImageAxes.XTick = [];
            app.ImageAxes.YTick = [];
            app.ImageAxes.Box = 'on';

            % Layout Variables
            controlLeft = 500;
            controlWidth = 350;
            baseHeight = 200; 

            % 1. NAME FIELD
            app.NameLabel = uilabel(app.UIFigure);
            app.NameLabel.Position = [controlLeft, baseHeight + 10, 80, 22];
            app.NameLabel.Text = 'Your Name:';
            app.NameLabel.FontWeight = 'bold';

            app.NameEdit = uieditfield(app.UIFigure, 'text');
            app.NameEdit.Position = [controlLeft + 90, baseHeight + 10, 150, 22];
            app.NameEdit.Placeholder = 'Enter Name';

            % 2. CHARACTER FIELD
            app.CharLabel = uilabel(app.UIFigure);
            app.CharLabel.Position = [controlLeft, baseHeight - 30, 120, 22];
            app.CharLabel.Text = 'Target Character:';
            app.CharLabel.FontWeight = 'bold';

            app.CharEdit = uieditfield(app.UIFigure, 'text');
            app.CharEdit.Position = [controlLeft + 130, baseHeight - 30, 60, 22];
            app.CharEdit.Value = 'A';
            app.CharEdit.HorizontalAlignment = 'center';
            app.CharEdit.FontSize = 14;

            % 3. START ID FIELD
            app.IDLabel = uilabel(app.UIFigure);
            app.IDLabel.Position = [controlLeft, baseHeight - 70, 120, 22];
            app.IDLabel.Text = 'Next Image ID:';
            app.IDLabel.FontWeight = 'bold';
            
            app.IDEdit = uieditfield(app.UIFigure, 'numeric');
            app.IDEdit.Position = [controlLeft + 130, baseHeight - 70, 60, 22];
            app.IDEdit.Value = 1; 
            app.IDEdit.Limits = [1 Inf];
            app.IDEdit.RoundFractionalValues = 'on';

            % 4. CAPTURE BUTTON
            app.CaptureButton = uibutton(app.UIFigure, 'push');
            app.CaptureButton.ButtonPushedFcn = createCallbackFcn(app, @CaptureButtonPushed, true);
            app.CaptureButton.Position = [controlLeft, baseHeight - 120, controlWidth, 40];
            app.CaptureButton.Text = 'Capture Frame';
            app.CaptureButton.FontSize = 16;
            app.CaptureButton.FontWeight = 'bold';
            app.CaptureButton.BackgroundColor = [0.85, 0.85, 0.85];

            % 5. INSTRUCTIONS
            app.Instruction = uilabel(app.UIFigure);
            app.Instruction.Position = [controlLeft, baseHeight - 150, controlWidth, 22];
            app.Instruction.Text = '(or press Spacebar - 0.25s delay)';
            app.Instruction.HorizontalAlignment = 'center';
            app.Instruction.FontColor = [0.4 0.4 0.4];

            % 6. COUNTER
            app.CountLabel = uilabel(app.UIFigure);
            app.CountLabel.Position = [controlLeft, baseHeight - 180, controlWidth, 22];
            app.CountLabel.Text = 'Session Count: 0';
            app.CountLabel.FontSize = 14;

            % 7. STATUS BAR
            app.StatusLabel = uilabel(app.UIFigure);
            app.StatusLabel.Position = [20 20 860 30];
            app.StatusLabel.Text = 'Enter Name & Start ID to begin...';
            app.StatusLabel.FontSize = 14;
            app.StatusLabel.BackgroundColor = [0.9 0.9 0.9];

            app.UIFigure.Visible = 'on';
        end
    end

    % App Creation and Deletion
    methods (Access = public)
        function app = data_collector
            createComponents(app)
            registerApp(app, app.UIFigure)
            app.CurrentCount = 0;
            runStartupFcn(app, @startupFcn)
        end

        function delete(app)
            delete(app.UIFigure)
        end
    end
end