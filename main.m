clc; clear all; close all;
% img = imread('A (1).png');
% img = rgb2gray(img);
% 
% points = detectSURFFeatures(img);
% 
% [f, vpts] = extractFeatures(img, points);
% figure(),imshow(img),hold on,plot(vpts);
 
%%
%pantechsolutions How to Extract Text from Images Using Matlab
faceDatabase = imageSet('FaceDatabase','recursive');
 
% % Display Montage of First Face
% figure(1);
% montage(faceDatabase(1).ImageLocation);
% title('Images of Single Face');
 
%%  Display Query Image and Database Side-Side
personToQuery = 1;
galleryImage = read(faceDatabase(personToQuery),1);
%figure(2);
for i=1:size(faceDatabase,2)    %size(fac)=[row,column];using"2" gives only column
    imageList(i) = faceDatabase(i).ImageLocation(4);
end
% subplot(1,2,1);imshow(galleryImage);
% subplot(1,2,2);montage(imageList);
% diff = zeros(1,9);
 
%% Split Database into Training & Test Sets
[training,test]= partition(faceDatabase,[.8 0.2]);
 
 
%% Extract and display Histogram of Oriented Gradient Features for single face
for person = 1:4;
    Inputimage = (read(training(person),3));
    if size(Inputimage,3)==3 % RGB image
 Inputimage=rgb2gray(Inputimage);
 disp(1)
end
    a=Inputimage;
    points = detectSURFFeatures(a);
    [f, vpts] = extractFeatures(a,points);
    
    
%     [hogFeature, visualization]= ...
%         extractHOGFeatures(read(training(person),3));
%     figure(1+person);
%     subplot(2,1,1);imshow(a);title('Input Face');
%     subplot(2,1,2);plot(vpts);title('SURF Feature');
end
%% Extract HOG Features for training set
%trainingFeatures = zeros(24*(size(training,2)*training(1).Count),64);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
       
        Inputimage = read(training(i),j);
    if size(Inputimage,3)==3 % RGB image
 Inputimage=rgb2gray(Inputimage);
 disp(1)
end
    a=Inputimage;
        points = detectSURFFeatures(a);
        trainingFeatures{i,:} = extractFeatures(a,points);
        trainingFeatures{i,:} = imresize(trainingFeatures{i,:},[48 64]);
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end
% 
%% Create 40 class classifier using fitcecoc
for i=1:length(trainingFeatures)
    faceClassifier{i} = fitcecoc(trainingFeatures{i},trainingLabel);
end
 
 
%% Read Image
[Inputimage map]=imread('Handwrite.jpg');
%% Show image
% figure(1)
% imshow(Inputimage);
% title('INPUT IMAGE WITH NOISE')
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
    n1=~Inputimage(min(r)-30:max(r)+30,min(c)-30:max(c)+30);
    
    n1 = imresize(n1, [128 128]);
%     figure(n+5)
%     imshow(n1);
    if n==1
        img=n1;
    end
end
 
% Test Images from Test Set
person=1;
%img = imread('1.PGM');
%img = n1;
%figure(121),imshow(n1)
 
points = detectSURFFeatures(img);
[f, vpts] = extractFeatures(img,points);
 
% [queryFeatures, visualization] = extractHOGFeatures(img);
% figure(6);
% subplot(2,1,1);imshow(img);title('Input Face');
% subplot(2,1,2);plot(vpts);title('SURF Feature');
 [personLabel,yci] = predict(faceClassifier{1,1},f);
if length(personLabel)>=46
personLabel = personLabel(46);
elseif length(trainingLabel)>length(personLabel)
    personLabel = personLabel(length(personLabel));
else
    personLabel = personLabel(length(trainingLabel));
end
%personLabel = personLabel(19);
%Map back to training set to find identity
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
figure(5)
subplot(1,2,1);imshow(img);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),3));title('Matched Class');
disp(personLabel);
