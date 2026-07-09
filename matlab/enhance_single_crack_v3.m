clear; clc;

[file, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'}, ...
    'Choose one crack image');

if isequal(file, 0)
    disp('No image selected.');
    return;
end

inputPath = fullfile(folder, file);
img = imread(inputPath);

if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end

denoised = medfilt2(gray, [3 3]);
enhanced = adapthisteq(denoised, "ClipLimit", 0.008, "NumTiles", [8 8]);

[~, name, ext] = fileparts(file);
outputPath = fullfile(folder, "MATLAB_" + name + ext);

imwrite(enhanced, outputPath);

figure;

subplot(1, 3, 1);
imshow(gray);
title('1. Grayscale');

subplot(1, 3, 2);
imshow(denoised);
title('2. Median Filter');

subplot(1, 3, 3);
imshow(enhanced);
title('3. CLAHE Final');

fprintf("Saved final MATLAB-enhanced image to:\n%s\n", outputPath);