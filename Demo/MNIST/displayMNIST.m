function varargout = displayMNIST(nets, dataset_folder)


if ( ~iscell(nets) )
    tmp{1}=nets;
    nets=tmp;
end


mOutputArgs     =   {};         % Variable for storing output when GUI returns
mIconCData      =   [];         % The icon CData edited by this GUI of dimension
                                % [mIconHeight, mIconWidth, 3]
mIsEditingIcon  =   false;      % Flag for indicating whether the current mouse 
                                % move is used for editing color or not
% Variables for supporting custom property/value pairs
mPropertyDefs   =   {...        % The supported custom property/value pairs of this GUI
                     'iconwidth',   @localValidateInput, 'mIconWidth';
                     'iconheight',  @localValidateInput, 'mIconHeight';
                     'MNISTfile',   @localValidateInput, 'mMNISTFile'};
mIconWidth      =   28;         % Use input property 'iconwidth' to initialize
mIconHeight     =   28;         % Use input property 'iconheight' to initialize
mMNISTFile      =   fullfile(dataset_folder,'t10k-images-idx3-ubyte'); %fullfile(matlabroot,'./'); 
Images = {0};
im_ptr = 1;

% Create all the UI objects in this GUI here so that they can
% be used in any functions in this GUI
hMainFigure     =   figure(...
                    'Units','characters',...
                    'MenuBar','none',...
                    'Toolbar','none',...
                    'Position',[71.8 34.7 106 36.15],...
                    'WindowStyle', 'normal',...
                    'WindowButtonDownFcn', @hMainFigureWindowButtonDownFcn,...
                    'WindowButtonUpFcn', @hMainFigureWindowButtonUpFcn,...
                    'WindowButtonMotionFcn', @hMainFigureWindowButtonMotionFcn);
hIconEditPanel  =    uipanel(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Clipping','on',...
                    'Position',[1.8 4.3 68.2 27.77]);
hIconEditAxes   =   axes(...
                    'Parent',hIconEditPanel,...
                    'vis','off',...
                    'Units','characters',...
                    'Position',[2 1.15 64 24.6]);
hIconFileText   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'HorizontalAlignment','left',...
                    'Position',[3 32.9 16.2 1.46],...
                    'String','MNIST file: ',...
                    'Style','text');

hIconFileEdit   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'HorizontalAlignment','left',...
                    'Position',[14.8 32.9 27.2 1.62],...
                    'String','Create a new icon or type in an icon image file for editing',...
                    'Enable','inactive',...
                    'Style','edit',...
                    'ButtondownFcn',@hIconFileEditButtondownFcn,...
                    'Callback',@hIconFileEditCallback);
hIconFileButton =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Callback',@hIconFileButtonCallback,...
                    'Position',[48 32.9 5.8 1.77],...
                    'String','...',...
                    'TooltipString','Import From Image File');

                

hIconNumberText   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'HorizontalAlignment','right',...
                    'Position',[65.8 32.9 17.2 1.62],...
                    'String','Image idx: ',...
                    'Style','text');
hIconNumEdit   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'HorizontalAlignment','left',...
                    'Position',[83.8 32.9 9.2 1.62],...
                    'String','1',...
                    'Style','edit');
hIconNumberButton =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Callback',@hIconNumButtonCallback,...
                    'Position',[95.8 32.9 7.2 1.62],...
                    'String','Set',...
                    'TooltipString','Set the desired Image idx');
                
                
                
hPreviewPanel   =   uipanel(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Title','Preview',...
                    'Clipping','on',...
                    'Position',[71.8 19.15 32.2 13]);
hPreviewControl =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','characters',...
                    'Enable','inactive',...
                    'Visible','off',...
                    'Position',[2 3.77 16.2 5.46],...
                    'String','');
hPreviewControlProcessed =   uicontrol(...
                    'Parent',hPreviewPanel,...
                    'Units','characters',...
                    'Enable','inactive',...
                    'Visible','off',...
                    'Position',[2 3.77 16.2 5.46],...
                    'String','');
                
hPrevDigitButton =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Position',[15 0.62 15 2.38],...
                    'String','<',...
                    'Callback',@hPrevDigitButtonCallback);

hNextDigitButton =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Position',[35 0.62 15 2.38],...
                    'String','>',...
                    'Callback',@hNextDigitButtonCallback);

               
hResultPanel   =    uipanel(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Title','Result',...
                    'Clipping','on',...
                    'Position',[71.8 4.3 32.2 13]);
hResultText =   uicontrol(...
                    'Parent',hResultPanel,...
                    'Units','normalized',...
                    'Style','text', ...
                    'Enable','inactive',...
                    'Visible','on',...
                    'FontSize', 50, ...
                    'Position',[.2 .2 .6 .6],...
                    'String','');
for i = 0:9,
    hResultDigits(1+i) = uicontrol(...
        'Parent',hResultPanel,...
        'Units','normalized',...
        'Enable','inactive',...
        'Style','text', ...
        'FontSize', 14, ...
        'Position',[.02+i*.095 .05 .09 .128],...
        'String', char('0'+i), ...
        'ForegroundColor', [1 1 1]);
end
                
hSectionLine    =   uipanel(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'HighlightColor',[0 0 0],...
                    'BorderType','line',...
                    'Title','',...
                    'Clipping','on',...
                    'Position',[2 3.62 102.4 0.077]);

hCancelButton   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Position',[85.8 0.62 17.8 2.38],...
                    'String','Exit',...
                    'Callback',@hCancelButtonCallback);
hClearButton   =   uicontrol(...
                    'Parent',hMainFigure,...
                    'Units','characters',...
                    'Position',[65.0 0.62 17.8 2.38],...
                    'String','Clear',...
                    'Callback',@hClearButtonCallback);
                
% Host the ColorPalette in the PaletteContainer and keep the function
% handle for getting its selected color for editing icon


% Make changes needed for proper look and feel and running on different
% platforms 
prepareLayout(hMainFigure);                            

% Process the command line input arguments supplied when the GUI is
% invoked 

% Initialize the iconEditor using the defaults or custom data given through
% property/value pairs
localUpdateIconPlot();

% Make the GUI on screen
set(hMainFigure,'visible', 'on');
movegui(hMainFigure,'onscreen');
figure(hMainFigure);
%hClearButtonCallback();
% Make the GUI blocking
%uiwait(hMainFigure);

% Return the edited icon CData if it is requested
mOutputArgs{1} =mIconCData;
if nargout>0
    [varargout{1:nargout}] = mOutputArgs{:};
end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonDownFcn(hObject, eventdata)
    % Callback called when mouse is pressed on the figure. Used to change
    % the color of the specific icon data point under the mouse to that of
    % the currently selected color of the colorPalette
        if (ancestor(gco,'axes') == hIconEditAxes)
            mIsEditingIcon = true;
            localEditColor();
        end
    end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonUpFcn(hObject, eventdata)
    % Callback called when mouse is release to exit the icon editing mode
        mIsEditingIcon = false;
    end

    %------------------------------------------------------------------
    function hMainFigureWindowButtonMotionFcn(hObject, eventdata)
    % Callback called when mouse is moving so that icon color data can be
    % updated in the editing mode
        if (ancestor(gco,'axes') == hIconEditAxes)
            localEditColor();
        end
    end

    %------------------------------------------------------------------
    function hIconFileEditCallback(hObject, eventdata)
    % Callback called when user has changed the icon file name from which
    % the icon can be loaded
        file = get(hObject,'String');
        if exist(file, 'file') ~= 2
            errordlg(['The given icon file cannot be found ' 10, file], ...
                    'Invalid Icon File', 'modal');
            set(hObject, 'String', mMNISTFile);
        else
            mIconCData = [];
            localUpdateIconPlot();            
        end
    end

    %------------------------------------------------------------------
    function hIconFileEditButtondownFcn(hObject, eventdata)
    % Callback called the first time the user pressed mouse on the icon
    % file editbox 
        set(hObject,'String','');
        set(hObject,'Enable','on');
        set(hObject,'ButtonDownFcn',[]);        
        uicontrol(hObject);
    end

    %------------------------------------------------------------------
    function hCancelButtonCallback(hObject, eventdata)
    % Callback called when the Cancel button is pressed
        mIconCData =[];
        uiresume;
        delete(hMainFigure);
    end
    %------------------------------------------------------------------
    function hRecognizeButtonCallback()
        
        image = preproc_image(mIconCData)';
        net_outs=0;
        for netIdx=1:length(nets)
            input = GetNetworkInputs(image , nets{netIdx} , 1);
            outs = feedForward(nets{netIdx}.layers, input , 1);
            net_outs=net_outs+outs{end}.activation;
        end
        net_outs = net_outs / length(nets);
        
       % outs = outs-min(outs);
       % outs = outs./sum(outs);
        
        %a{end}.activation
        [M,m] = max(net_outs);

        max_out = M;
        for out_i = 1:numel(net_outs),
            set(hResultDigits(out_i), 'ForegroundColor', [1 1 1]*(1.8-net_outs(out_i))/3.6)
        end
        
        digit = m - 1;

        if (M>0.2)
            set(hResultText, 'string', char('0'+digit), 'ForegroundColor', [1 1 1]*(1.8-max_out)/3.6)
        else
            set(hResultText, 'string', '?', 'ForegroundColor', [1 1 1]*(1.8-max_out)/3.6)
        end
        
        set(hResultPanel, 'Title',['Result:' num2str(digit), ',Conf=' num2str(M*100,4) '%']);
        
        
    end
    %------------------------------------------------------------------
    function hClearButtonCallback(hObject, eventdata)
        mIconCData = ones(mIconHeight, mIconWidth, 3);
        localUpdateIconPlot();
        hRecognizeButtonCallback()
    end

    %------------------------------------------------------------------
    function hIconNumButtonCallback(hObject, eventdata)
    % Callback called when the icon file selection button is pressed
        imgIdx = str2double(get(hIconNumEdit, 'String'));  
        
        im_ptr = 1+mod(imgIdx-1,numel(Images));

        im = abs(double(Images{im_ptr})/255-1)';
        mIconCData = cat(3,im,im,im);    
        localUpdateIconPlot();
        hRecognizeButtonCallback()

    end


    %------------------------------------------------------------------
    function hIconFileButtonCallback(hObject, eventdata)
    % Callback called when the icon file selection button is pressed
        filespec = {'*.*', 'Database file '};
        [filename, pathname] = uigetfile(filespec, 'Pick an database file', mMNISTFile);

        if ~isequal(filename,0)
            mMNISTFile = fullfile(pathname, filename);             
            set(hIconFileEdit, 'ButtonDownFcn',[]);            
            set(hIconFileEdit, 'Enable','on');            
            set(hIconNumEdit, 'Enable','on');            
            
            mIconCData = [];
            localUpdateIconPlot();            
            
        elseif isempty(mIconCData)
            set(hPreviewControl,'Visible', 'off');            
        end
    end

    function hPrevDigitButtonCallback(hObject, eventdata)
        if(im_ptr>1)
            im_ptr=im_ptr-1;
        end
        im = abs(double(Images{im_ptr})/255-1)';
        mIconCData = cat(3,im,im,im);   
        localUpdateIconPlot();
        hRecognizeButtonCallback()
    end

    function hNextDigitButtonCallback(hObject, eventdata)
        if(im_ptr<numel(Images))
            im_ptr=im_ptr+1;
        end
        im = abs(double(Images{im_ptr})/255-1)';
        mIconCData = cat(3,im,im,im);    
        localUpdateIconPlot();
        hRecognizeButtonCallback()
    end
    %------------------------------------------------------------------
    function localEditColor
    % helper function that changes the color of an icon data point to
    % that of the currently selected color in colorPalette 
        if mIsEditingIcon
            pt = get(hIconEditAxes,'currentpoint');

            x = max(1, min(ceil(pt(1,1)), mIconWidth));
            y = max(1, min(ceil(pt(1,2)), mIconHeight));

            % update color of the selected block
            m = get(gcf,'SelectionType');
            if m(1) == 'n', % left button pressed
                %fprintf('Updateing %d,%d to 0\n',y,x);
                
                mIconCData(y, x,:) = 0;
                if y<mIconHeight,   mIconCData(y+1,x,:) = .8*mIconCData(y+1,x,:); end
                if x<mIconWidth,    mIconCData(y,x+1,:) = .8*mIconCData(y,x+1,:); end
                if y>1,             mIconCData(y-1,x,:) = .8*mIconCData(y-1,x,:); end
                if x>1,             mIconCData(y,x-1,:) = .8*mIconCData(y,x-1,:); end
            else
                mIconCData(y, x,:) = 1;
                %fprintf('Updateing %d,%d to 1\n',y,x);
            end
            localUpdateIconPlot();
        end
    end

    %------------------------------------------------------------------
    function localUpdateIconPlot   
    % helper function that updates the iconEditor when the icon data
    % changes
        %initialize icon CData if it is not initialized
        set(hPreviewPanel,'Title',['Preview, image idx=' num2str(im_ptr)]);
        if isempty(mIconCData)
            if exist(mMNISTFile, 'file') == 2
                try
                    Images = readMNIST_image(mMNISTFile,60000);
                    %im_ptr = 1;
                    im = abs(double(Images{im_ptr})/255-1)';
                    mIconCData = cat(3,im,im,im);
                    set(hIconFileEdit, 'String', mMNISTFile);            
                catch
                    errordlg(['Could not load MNIST database file successfully. ',...
                              'Make sure the file name is correct: ' mMNISTFile],...
                              'Invalid MNIST File', 'modal');
                    mIconCData = nan(mIconHeight, mIconWidth, 3);
                end
            else 
                mIconCData = nan(mIconHeight, mIconWidth, 3);
            end
        end
        
        % update preview control
        rows = size(mIconCData, 1);
        cols = size(mIconCData, 2);
        previewSize = getpixelposition(hPreviewPanel);
        % compensate for the title
        previewSize(4) = previewSize(4) -15;
        controlWidth = previewSize(3);
        controlHeight = previewSize(4);  
        controlMargin = 6;
        if rows+controlMargin<controlHeight
            controlHeight = rows+controlMargin;
        end
        if cols+controlMargin<controlWidth
            controlWidth = cols+controlMargin;
        end        
        setpixelposition(hPreviewControl,[(previewSize(3)-2*controlWidth)/3,(previewSize(4)-controlHeight)/3*2, controlWidth, controlHeight]); 
        
        iconCData = mIconCData;
        image = preproc_image(mIconCData)';
        cm = round(centerOfMass(image-min(min(image))))+1;
        iconCData(cm(1),cm(2),:) = cat(3,1,0,0);
        set(hPreviewControl,'CData', iconCData,'Visible','on');

        setpixelposition(hPreviewControlProcessed,[(previewSize(3)-2*controlWidth)/3*2+controlWidth,(previewSize(4)-controlHeight)/3*2, controlWidth, controlHeight]); 
        
        input = GetNetworkInputs(image , nets{1}  ,1 );
        input = imresize(input,[size(mIconCData,1) size(mIconCData,2)]);
        input = input-min(min(input));
        maxIm=max(max(input));
        if (maxIm~=0)
            input = input/maxIm;
        end
        input = 1-input;

        colimage = cat(3, input, input, input);
        set(hPreviewControlProcessed,'CData', colimage,'Visible','on');

        % update icon edit pane
        set(hIconEditPanel, 'Title',['Icon Edit Pane (', num2str(rows),' X ', num2str(cols),')']);
        
        s = findobj(hIconEditPanel,'type','surface');        
        if isempty(s)
            gridColor = get(0, 'defaultuicontrolbackgroundcolor') - 0.2;
            gridColor(gridColor<0)=0;
            s=surface('edgecolor',gridColor,'parent',hIconEditAxes);
        end        
        %set xdata, ydata, zdata in case the rows and/or cols change
        set(s,'xdata',0:cols,'ydata',0:rows,'zdata',zeros(rows+1,cols+1),'cdata',localGetIconCDataWithNaNs());

        set(hIconEditAxes,'xlim',[-.5 cols+.5],'ylim',[-.5 rows+.5]);
        axis(hIconEditAxes, 'ij', 'off');    
        
        hRecognizeButtonCallback();
    end

    %------------------------------------------------------------------
	function cdwithnan = localGetIconCDataWithNaNs()
		% Add NaN to edge of mIconCData so the entire icon renders in the
		% drawing pane.  This is necessary because of surface behavior.
		cdwithnan = mIconCData;
		cdwithnan(:,end+1,:) = NaN;
		cdwithnan(end+1,:,:) = NaN;
		
	end

    %------------------------------------------------------------------
    function isValid = localValidateInput(property, value)
    % helper function that validates the user provided input property/value
    % pairs. You can choose to show warnings or errors here.
        isValid = false;
        switch lower(property)
            case {'iconwidth', 'iconheight'}
                if isnumeric(value) && value >0
                    isValid = true;
                end
            case 'MNISTfile'
                if exist(value,'file')==2
                    isValid = true;                    
                end
        end
    end
end % end of iconEditor

%------------------------------------------------------------------
function prepareLayout(topContainer)
% This is a utility function that takes care of issues related to
% look&feel and running across multiple platforms. You can reuse
% this function in other GUIs or modify it to fit your needs.
    allObjects = findall(topContainer);
    warning off  %Temporary presentation fix
    try
        titles=get(allObjects(isprop(allObjects,'TitleHandle')), 'TitleHandle');
        allObjects(ismember(allObjects,[titles{:}])) = [];
    catch
    end
    warning on

    % Use the name of this GUI file as the title of the figure
    defaultColor = get(0, 'defaultuicontrolbackgroundcolor');
    if isa(handle(topContainer),'figure')
        set(topContainer,'Name', mfilename, 'NumberTitle','off');
        % Make figure color matches that of GUI objects
        set(topContainer, 'Color',defaultColor);
    end

    % Make GUI objects available to callbacks so that they cannot
    % be changes accidentally by other MATLAB commands
    set(allObjects(isprop(allObjects,'HandleVisibility')), 'HandleVisibility', 'Callback');

    % Make the GUI run properly across multiple platforms by using
    % the proper units
    if strcmpi(get(topContainer, 'Resize'),'on')
        set(allObjects(isprop(allObjects,'Units')),'Units','Normalized');
    else
        set(allObjects(isprop(allObjects,'Units')),'Units','Characters');
    end

    % You may want to change the default color of editbox,
    % popupmenu, and listbox to white on Windows 
    if ispc
        candidates = [findobj(allObjects, 'Style','Popupmenu'),...
                           findobj(allObjects, 'Style','Edit'),...
                           findobj(allObjects, 'Style','Listbox')];
        set(findobj(candidates,'BackgroundColor', defaultColor), 'BackgroundColor','white');
    end
end
function out = preproc_image(id)
    %Preprocess single image
    Inorm = id(:,:,1);
    Inorm(~isfinite(Inorm)) = 1;
    Inorm = abs(Inorm-1)';
    out = zeros(32);
    out(3:30,3:30) = Inorm;
    if sum(out(:))>0,
        out = reshape(mapstd(out(:)'), 32, 32);
    end
    
end
function I = readMNIST_image(filepath,num)
    %readMNIST_image MNIST handwriten image database reading. Reads only images
    %without labels, with specified filename
    %
    %  Syntax
    %  
    %    I = readMNIST_image(filepath,num)
    %    
    %  Description
    %   Input:
    %    filepath - name of database file with path
    %    n - number of images to process
    %   Output:
    %    I - cell array of training images 28x28 size
    %
    %(c) Sirotenko Mikhail, 2009
    %===========Loading training set

    fid = fopen(filepath,'r','b');  %big-endian
    magicNum = fread(fid,1,'int32');    %Magic number
    if(magicNum~=2051) 
        display('Error: cant find magic number');
        return;
    end
    imgNum = fread(fid,1,'int32');  %Number of images
    rowSz = fread(fid,1,'int32');   %Image height
    colSz = fread(fid,1,'int32');   %Image width

    if(num<imgNum) 
        imgNum=num; 
    end

    for k=1:imgNum
        I{k} = uint8(fread(fid,[rowSz colSz],'uchar'));
    end
    fclose(fid);
end
