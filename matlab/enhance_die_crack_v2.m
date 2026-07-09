% Enhance DIE_CRACK images - Version 2.
% This version is intentionally milder than enhance_die_crack.m:
% grayscale -> median denoise -> light CLAHE -> very mild sharpening.
% It saves a new dataset and copies YOLO labels unchanged.

clear; clc;

srcRoot = fullfile(pwd, "ChipDetection_single_class", "DIE_CRACK");
dstRoot = fullfile(pwd, "ChipDetection_single_class_enhanced_v2", "DIE_CRACK");

splits = ["train", "valid", "test"];

if ~isfolder(srcRoot)
    error("Source folder not found: %s", srcRoot);
end

for s = 1:numel(splits)
    splitName = splits(s);

    srcImageDir = fullfile(srcRoot, splitName, "images");
    srcLabelDir = fullfile(srcRoot, splitName, "labels");
    dstImageDir = fullfile(dstRoot, splitName, "images");
    dstLabelDir = fullfile(dstRoot, splitName, "labels");

    if ~isfolder(srcImageDir)
        warning("Skipping missing image folder: %s", srcImageDir);
        continue;
    end

    if ~isfolder(dstImageDir)
        mkdir(dstImageDir);
    end
    if ~isfolder(dstLabelDir)
        mkdir(dstLabelDir);
    end

    imageFiles = [ ...
        dir(fullfile(srcImageDir, "*.jpg")); ...
        dir(fullfile(srcImageDir, "*.jpeg")); ...
        dir(fullfile(srcImageDir, "*.png")) ...
    ];

    fprintf("Enhancing %s images with V2: %d files\n", splitName, numel(imageFiles));

    for i = 1:numel(imageFiles)
        imageName = imageFiles(i).name;
        srcImagePath = fullfile(srcImageDir, imageName);
        dstImagePath = fullfile(dstImageDir, imageName);

        img = imread(srcImagePath);

        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end

        % V2 crack enhancement:
        % 1. Median filter reduces small speckle noise.
        % 2. Light CLAHE improves crack contrast without over-amplifying texture.
        % 3. Mild sharpening keeps crack edges visible but avoids harsh artifacts.
        denoised = medfilt2(gray, [3 3]);
        contrast = adapthisteq(denoised, "ClipLimit", 0.006, "NumTiles", [8 8]);
        enhanced = imsharpen(contrast, "Radius", 0.7, "Amount", 0.35);

        imwrite(enhanced, dstImagePath);

        [~, baseName, ~] = fileparts(imageName);
        srcLabelPath = fullfile(srcLabelDir, baseName + ".txt");
        dstLabelPath = fullfile(dstLabelDir, baseName + ".txt");

        if isfile(srcLabelPath)
            copyfile(srcLabelPath, dstLabelPath);
        end
    end
end

dataYamlPath = fullfile(dstRoot, "data.yaml");
fid = fopen(dataYamlPath, "w");
fprintf(fid, "path: %s\n", strrep(dstRoot, "\", "/"));
fprintf(fid, "train: train/images\n");
fprintf(fid, "val: valid/images\n");
fprintf(fid, "test: test/images\n");
fprintf(fid, "nc: 1\n");
fprintf(fid, "names:\n");
fprintf(fid, "  - DIE_CRACK\n");
fclose(fid);

fprintf("\nDone.\n");
fprintf("Enhanced V2 dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
