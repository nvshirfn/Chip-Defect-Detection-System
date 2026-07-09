% Enhance DIE_CRACK images from the single_class_v2 split.
% Same technique as enhance_die_crack_v3.m (median denoise -> light CLAHE,
% no sharpening), applied to the properly-split single_class_v2 dataset
% instead of single_class_raw (real-only valid/test, synthetic images in
% train only, plus background images). Saves a new dataset and copies
% YOLO labels unchanged, including empty labels for background images.

clear; clc;

srcRoot = fullfile(pwd, "single_class_v2", "DIE_CRACK");
dstRoot = fullfile(pwd, "crack_v2split_enhanced_matlab", "DIE_CRACK");

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

    fprintf("Enhancing %s images with V2-split CLAHE: %d files\n", splitName, numel(imageFiles));

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

        % Same crack enhancement as V3: median filter reduces speckle
        % noise, then light CLAHE improves local crack contrast. No
        % sharpening, to reduce false edge amplification.
        denoised = medfilt2(gray, [3 3]);
        enhanced = adapthisteq(denoised, "ClipLimit", 0.008, "NumTiles", [8 8]);

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
fprintf("Enhanced V2-split dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
