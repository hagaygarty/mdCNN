function [ CIFAR10 ] = getCIFAR10data( dataset_folder )
% this function will download the CIFAR10 dataset if not exist already.
% after downloading it will then parse the raw files and create a CIFAR10.mat file
% containning the test/train images and labels.
% function returns a struct containing the images+labels.
% I,labels,I_test,labels_test. Every element is an array containing the images/labels

outFile = fullfile(dataset_folder ,'CIFAR10.mat');

if (~exist(outFile,'file'))
    url='http://www.cs.toronto.edu/~kriz/';
    files = {'cifar-10-matlab.tar.gz'};

    fprintf('Preparing CIFAR10.mat file since it does not exist. (done only once)\n');

    for fileIdx=1:numel(files)
        [~,fname,~] = fileparts(files{fileIdx});
        if ( exist(fullfile(dataset_folder,fname),'file'))
            continue;
        end
        fprintf('Downloading file %s from %s .. this may take a while',files{fileIdx},url);
        gunzip([url files{fileIdx}], dataset_folder);
        untar(fullfile(dataset_folder ,'cifar-10-matlab.tar'),dataset_folder);
        fprintf('Done\n');
    end

    parseCIFARfiles(fullfile(dataset_folder, 'cifar-10-batches-mat'),outFile);
end

fprintf('Loading CIFAR10 mat file\n');

CIFAR10 = load(outFile);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = parseCIFARfiles(path,outFile)

fprintf('Parsing CIFAR10 dataset\n');

load(fullfile(path, 'batches.meta.mat')); %include the labels first

sources=dir(fullfile(path, 'data_batch_*.mat'));
imageIdx=1;
for i=1:length(sources)
    fileData = load(fullfile(path,sources(i).name));
    for j=1:length(fileData.labels)
        I{imageIdx} = permute(reshape(fileData.data(j,:),32,32,3),[2 1 3]);
        labels(imageIdx) = fileData.labels(j);  
        imageIdx = imageIdx+1;
    end
end

sources=dir(fullfile(path, 'test_batch.mat'));
imageIdx=1;
for i=1:length(sources)
    fileData = load(fullfile(path, sources(i).name));
    for j=1:length(fileData.labels)
        I_test{imageIdx} = permute(reshape(fileData.data(j,:),32,32,3),[2 1 3]);
        labels_test(imageIdx) = fileData.labels(j);  
        imageIdx = imageIdx+1;
    end
end


save(outFile,'label_names','I','labels','I_test','labels_test');

end
