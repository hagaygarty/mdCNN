function [ images ] = loadBackroundImages(  )
path = 'BackroundImages';
files=dir(path);
idx=1;
for i=1:length(files)
    if (files(i).isdir==1)
        continue;
    end
    image = imread([path '\' files(i).name]);
    if ( size(image,3) ~=1)
        image=rgb2gray(image);
    end
    
    images{idx}=imresize(image,[256 256],'nearest','Antialiasing',true);
    idx=idx+1;
end

%save('images.mat','images');

end

