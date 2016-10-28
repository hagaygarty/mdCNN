function [ MNIST ] = getMNIST3Ddata( dst_folder )
% this function will download the MNIST dataset if not exist already.
% after downloading it will then parse the files and create a MNIST.mat file
% containing the test/train images and labels.
% function returns a struct containing the images+labels
% I,labels,I_test,labels_test , every element in the struct is an array containing the images/labels


outFile = fullfile(dst_folder ,'MNIST3d.mat');

if (~exist(outFile,'file'))
    url='http://yann.lecun.com/exdb/mnist/';
    files = {'t10k-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz' , 'train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz'};

    fprintf('Preparing MNIST.mat file since it does not exist. (done only once)\n');

    for fileIdx=1:numel(files)
        [~,fname,~] = fileparts(files{fileIdx});
        if ( exist(fullfile(dst_folder,fname),'file'))
            continue;
        end
        fprintf('Downloading file %s from %s  ...',files{fileIdx},url);
        gunzip([url files{fileIdx}], dst_folder);
        fprintf('Done\n');
    end

    parseMNIST3Dfiles(dst_folder,outFile);
end

MNIST = load(outFile);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [res] = Transform2dto3d(image,len)
    res = zeros([size(image) size(image,1)], 'uint8');
    for z=0:(len-1)
        SE = strel('diamond',len-z);
        deIm = imdilate(image,SE,'same');
        edgeIm = uint8(255*edge(deIm));
        res(:,:, round(size(res,3)/2)+z) = edgeIm;
        res(:,:, round(size(res,3)/2)-z) = edgeIm;
    end
    z=len;
    res(:,:, round(size(res,3)/2)+z) = image;
    res(:,:, round(size(res,3)/2)-z) = image;
end

function [] = parseMNIST3Dfiles(path,outFile)
%readMNIST MNIST handwriten image database reading.
%   Output:
%    I - cell array of training images 28x28 size
%    labels - vector of labels (true digits) for training set
%    I_test - cell array of testing images 28x28 size
%    labels_test - vector of labels (true digits) for testing set

if (exist('path','var')==0)
    path = './';
end

fact=1;
len=floor(12*fact);
len=3;

fprintf('Prepering MNIST3d dataset... (done only once)\n');

if(~exist(fullfile(path ,'train-images-idx3-ubyte'),'file'))
    error('Training set of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
end
fid = fopen(fullfile(path ,'train-images-idx3-ubyte'),'r','b');  %big-endian
magicNum = fread(fid,1,'int32');    %Magic number
if(magicNum~=2051) 
    display('Error: cant find magic number');
    return;
end
imgNum = fread(fid,1,'int32');  %Number of images
rowSz = fread(fid,1,'int32');   %Image height
colSz = fread(fid,1,'int32');   %Image width


for k=1:imgNum
    I{k} = uint8(fread(fid,[rowSz colSz],'uchar'))';
    I{k} = imresize(I{k} , fact,'bilinear','Antialiasing',true );
    I{k} = Transform2dto3d(I{k},len);
    if ( mod(k,5000)==0)
       % close all;
       % showIso(I{k},[]);
        fprintf('Finish preparing image %d of %d\n',k,imgNum);
    end
end

fclose(fid);

%============Loading labels
if(~exist(fullfile(path, 'train-labels-idx1-ubyte') ,'file'))
    error('Training labels of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
end
fid = fopen(fullfile(path ,'train-labels-idx1-ubyte'),'r','b');  %big-endian
magicNum = fread(fid,1,'int32');    %Magic number
if(magicNum~=2049) 
    display('Error: cant find magic number');
    return;
end
itmNum = fread(fid,1,'int32');  %Number of labels

labels = uint8(fread(fid,itmNum,'uint8'));   %Load all labels

fclose(fid);

%============All the same for test set
if(~exist(fullfile(path, 't10k-images-idx3-ubyte'),'file'))
    error('Test images of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
end

fid = fopen(fullfile(path, 't10k-images-idx3-ubyte'),'r','b');  
magicNum = fread(fid,1,'int32');    
if(magicNum~=2051) 
    display('Error: cant find magic number');
    return;
end
imgNum = fread(fid,1,'int32');  
rowSz = fread(fid,1,'int32');   
colSz = fread(fid,1,'int32');   


for k=1:imgNum
    I_test{k} = uint8(fread(fid,[rowSz colSz],'uchar'))';
    I_test{k} = imresize(I_test{k} , fact,'bilinear','Antialiasing',true );
    I_test{k} = Transform2dto3d(I_test{k},len);
   
    if ( mod(k,5000)==0)
       % close all;
       % showIso(I{k},[]);
        fprintf('Finish image (test) %d of %d\n',k,imgNum);
    end
end

fclose(fid);

%============Test labels
if(~exist(fullfile(path, 't10k-labels-idx1-ubyte'),'file'))
    error('Test labels of MNIST not found. Please download it from http://yann.lecun.com/exdb/mnist/ and put to ./MNIST folder');
end

fid = fopen(fullfile(path, 't10k-labels-idx1-ubyte'),'r','b');  
magicNum = fread(fid,1,'int32');    
if(magicNum~=2049) 
    display('Error: cant find magic number');
    return;
end
itmNum = fread(fid,1,'int32');  


labels_test = uint8(fread(fid,itmNum,'uint8'));   

fclose(fid);

labels = fixErrorsInMNIST(labels);

save(outFile,'I','labels','I_test','labels_test','-v7.3');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ labels ] = fixErrorsInMNIST( labels )
% some clear errors found on original MnisT data set. The below corrects
% the labels
errorsIdx       = [59916 10995 26561 32343 43455 45353];
correctLabels   = [ 7     9     1     7     3      1];

for idx=1:length(errorsIdx)
    labels(errorsIdx(idx)) = correctLabels(idx);
end

end

