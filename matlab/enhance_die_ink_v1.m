% Enhance DIE_INK images for CSC566 project.
% Ink defects usually appear as dark stains/blobs, so this version uses:
% grayscale -> median filter -> contrast adjustment -> bottom-hat dark feature enhancement.
% It saves a new YOLO dataset and copies labels unchanged.

clear; clc;

srcRoot = fullfile(pwd, "ChipDetection_single_class", "DIE_INK");
dstRoot = fullfile(pwd, "ChipDetection_single_class_enhanced_ink_v1", "DIE_INK");

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

    fprintf("Enhancing %s DIE_INK images: %d files\n", splitName, numel(imageFiles));

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

        % Bottom-hat emphasizes dark features. Subtracting it makes dark
        % ink regions slightly stronger while keeping chip context visible.
        darkFeatures = imbothat(adjusted, strel("disk", 8));
        enhanced = imsubtract(adjusted, darkFeatures);
        enhanced = imadjust(enhanced);

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
fprintf("Enhanced DIE_INK dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
