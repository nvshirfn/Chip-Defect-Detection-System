% Enhance DIE_INK images from the single_class_v2 split.
% Same technique as enhance_die_ink_v3.m (bottom-hat dark-feature
% isolation + light CLAHE, the best-performing ink pipeline so far),
% applied to the properly-split single_class_v2 dataset instead of
% single_class_raw (real-only valid/test, synthetic images in train
% only, plus background images). Saves a new dataset and copies YOLO
% labels unchanged, including empty labels for background images.

clear; clc;

srcRoot = fullfile(pwd, "single_class_v2", "DIE_INK");
dstRoot = fullfile(pwd, "ink_v2split_enhanced_matlab", "DIE_INK");

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

    fprintf("Enhancing %s DIE_INK images with V2-split Bottom-Hat+CLAHE: %d files\n", splitName, numel(imageFiles));

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

        denoised = medfilt2(gray, [3 3]);
        adjusted = imadjust(denoised);

        % Bottom-hat isolates dark ink stains against their local
        % background, then light CLAHE sharpens local contrast on the
        % isolated result -- same as V3.
        darkFeatures = imbothat(adjusted, strel("disk", 8));
        subtracted = imsubtract(adjusted, darkFeatures);
        enhanced = adapthisteq(subtracted, "ClipLimit", 0.008, "NumTiles", [8 8]);

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
fprintf(fid, "  - DIE_INK\n");
fclose(fid);

fprintf("\nDone.\n");
fprintf("Enhanced V2-split DIE_INK dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
