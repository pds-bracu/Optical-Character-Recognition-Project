%pantechsolutions How to Extract Text from Images Using Matlab
clc; clear all; close all;
PGM_updated = imageSet('PGM_updated','recursive');
 
% % Display Montage of First Face
% figure(1);
% montage(faceDatabase(1).ImageLocation);
% title('Images of Single Face');
 
%%  Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(PGM_updated(personToQuery),1);
%figure(2);
for i=1:size(PGM_updated,2)    %size(fac)=[row,column];using"2" gives only column
    imageList(i) = PGM_updated(i).ImageLocation(4);
end
% subplot(1,2,1);imshow(galleryImage);
% subplot(1,2,2);montage(imageList);
% diff = zeros(1,9);
 
%% Split Database into Training & Test Sets
[training,test]= partition(PGM_updated,[0.75 0.25]);
 
 
%% Extract and display Histogram of Oriented Gradient Features for single face
for person = 1:8;
    [hogFeature, visualization]= ...
        extractHOGFeatures(read(training(person),3));
%     figure(1+person);
%     subplot(2,1,1);imshow(read(training(person),3));title('Input Face');
%     subplot(2,1,2);plot(visualization);title('HoG Feature');
end
%% Extract HOG Features for training set
trainingFeatures = zeros(size(training,2)*training(1).Count,8100);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end
% 
%% Create 40 class classifier using fitcecoc
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
 
%% Read Image
[Inputimage map]=imread('Handwrite.jpg');

%% Convert to gray scale
if size(Inputimage,3)==3  % RGB image
    Inputimage=rgb2gray(Inputimage);
    %disp(1)
end
%% Convert to binary image
 
threshold = graythresh(Inputimage);
%BW = ~imbinarize(Inputimage,'adaptive','ForegroundPolarity','dark','Sensitivity',.5);
BW=~im2bw(Inputimage,map,0.4);
Inputimage = BW;
figure()
imshow(BW)
title('Binary Version of Image')%%%%%make it white letters on black background
%% Remove all object containing fewer than 30 pixels
Inputimage = bwareaopen(Inputimage,30);

%% Label connected components
[L Ne]=bwlabel(Inputimage);
propied=regionprops(L,'BoundingBox');

%% Objects extraction
 
for n=1:Ne
    %n = 1:Ne
    [r,c] = find(L==n);
    n1=~Inputimage(min(r)-60:max(r)+60,min(c)-60:max(c)+60);
    
    n1 = imresize(n1, [128 128]);
    %figure(n+5)
    %imshow(n1);
    
        img=n1;
    
    [queryFeatures, visualization] = extractHOGFeatures(img);
 
personLabel = predict(faceClassifier,queryFeatures);
%Map back to training set to find identity
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
figure(5+n)
subplot(1,2,1);imshow(img);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),3));title('Matched Class');disp(personLabel);
end
