function [nucleiIndexes, BW] = NucleiDetection_HE( RGB, MinPixel, MaxPixel, splitNuclei )

% REGION OF INTEREST
adjustRed = imadjust(RGB(:,:,1));
roiGamma = imadjust(adjustRed,[0;0.5]);
roiMaskThresh = roiGamma < 250;
roiMaskFill = bwareaopen(~roiMaskThresh,MinPixel);
roiMaskNoiseRem = bwareaopen(~roiMaskFill,150);
roiMaskDilat =  imdilate(roiMaskNoiseRem,strel('disk',3));
roiMask = imfill(roiMaskDilat,'holes');

disp('Second Detection Method: ');

hsv = rgb2hsv_fast(RGB);
hsv(:,:,3) = 0.8;
RGB2 = uint8(hsv2rgb(hsv).*255);
diffRGB = RGB2-RGB;
adjRGB = imadjust(diffRGB,[0 0 0; 0.4 0.4 0.4],[]);
%figure, imshow(adjRGB)

gmask = fspecial('gauss',30,3);
gauss = imfilter(adjRGB(:,:,3),gmask);

bw1 = gauss>100;
bw1 = bw1.*roiMask;
bw2 = imfill(bwareaopen(bw1,MinPixel),'holes');
%bw2 = imopen(bw2, strel('disk', 5));
%figure, imshow(bw2);

if(splitNuclei == 1)
    [L,~] = splitCells(RGB, bw2, 20, 100, 0.95, 1, 1);
else
    L = bwlabel(bw2);
end

R = regionprops(L,'Area');
ind = find([R.Area] >= MinPixel & [R.Area] <= MaxPixel);
L = ismember(L,ind);
bw4 = L > 0;
%figure, imshow(bw4);

BW = bw4;

s = regionprops(L,'Centroid');
centroids = cat(1, s.Centroid);
nucleiIndexes = [];

if size(centroids) > 0
    disp('Centroids found');
    % CENTROID DETECTION
    indexY = uint16(centroids(:,2));
    indexX = uint16(centroids(:,1));
    %figure, imshow(RGB,[]), hold on, plot(indexX,indexY,'+b')

    % DISTANCE TRANSFORM
    bwDist = bwdist(~bw2);
    gaussElmt = fspecial('Gaussian',15,5);
    filtDist = imfilter(bwDist,gaussElmt);

    % ORDERING CANDIDATES BY LOG CRITERIA = CENTER OF CELL NUCLEI CONSTRAINT
    scoreP = diag(filtDist(indexY,indexX));
    logMap = imfilter(medfilt2(filtDist,[7 7]),fspecial('log',5,2));
    logCriteria = abs(logMap)+roiMask;
    scoreLog = diag(logCriteria(indexY,indexX));

    % ORDERING CANDIDATES BY CENTER & SIZE CRITERIA
    % chartCombine = [scoreP scoreLog];
    scoreCombine = (scoreP)./(scoreLog);
    rawNucleiIndexes = [round(scoreCombine) indexX indexY];
    nucleiIndexes = sortrows(rawNucleiIndexes,-1);
end
end

