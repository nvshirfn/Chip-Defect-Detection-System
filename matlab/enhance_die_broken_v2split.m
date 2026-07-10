% Enhance DIE_BROKEN images from the single_class_v2 split.
% DIE_BROKEN defects are structural (chipped/fractured edges), not thin
% lines (crack) or dark blobs (ink), so this pipeline emphasizes
% edge/boundary definition instead of just local contrast, mirroring
% the tuned Python pipeline (enhance_die_broken_v2split_python.py):
% grayscale -> median denoise -> mild CLAHE -> light unsharp masking.

clear; clc;

srcRoot = fullfile(pwd, "single_class_v2", "DIE_BROKEN");
dstRoot = fullfile(pwd, "broken_v2split_enhanced_matlab", "DIE_BROKEN");

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

    fprintf("Enhancing %s DIE_BROKEN images with V2-split CLAHE+Unsharp: %d files\n", splitName, numel(imageFiles));

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

        % Same parameters as the tuned Python version: median denoise,
        % mild CLAHE, then a light unsharp mask to emphasize fracture
        % edges without amplifying noise.
        denoised = medfilt2(gray, [3 3]);
        contrast = adapthisteq(denoised, "ClipLimit", 0.006, "NumTiles", [8 8]);
        enhanced = imsharpen(contrast, "Radius", 1.5, "Amount", 0.15);

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
fprintf(fid, "  - DIE_BROKEN\n");
fclose(fid);

fprintf("\nDone.\n");
fprintf("Enhanced V2-split DIE_BROKEN dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
