% Enhance DIE_INK images - Version 4 (ROI-segmented).
% DIE_INK had the highest false-positive rate against background of any
% class (60% of all background false positives were mislabeled as ink).
% V1-V3 only changed contrast within the whole image; V4 additionally
% segments the die region (light square) from the background grid (dark
% pads) via Otsu thresholding, then masks the background to a neutral
% fill so the model never sees the confusing texture at all. Keeps the
% V3 bottom-hat + CLAHE technique on top, since that was the
% best-performing contrast pipeline so far.

clear; clc;

srcRoot = fullfile(pwd, "single_class_v2", "DIE_INK");
dstRoot = fullfile(pwd, "ink_v2split_enhanced_v4_matlab", "DIE_INK");

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

    fprintf("Enhancing %s DIE_INK images with V4 ROI-segmented: %d files\n", splitName, numel(imageFiles));

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

        % Same V3 bottom-hat + CLAHE contrast pipeline.
        darkFeatures = imbothat(adjusted, strel("disk", 8));
        subtracted = imsubtract(adjusted, darkFeatures);
        enhanced = adapthisteq(subtracted, "ClipLimit", 0.008, "NumTiles", [8 8]);

        % V4 addition: segment the die region via Otsu thresholding on the
        % denoised image, keep only the largest connected bright
        % component (the die), and mask everything else out.
        level = graythresh(denoised);
        binary = imbinarize(denoised, level);
        cleaned = imclose(binary, strel("disk", 4));

        cc = bwconncomp(cleaned, 8);
        if cc.NumObjects == 0
            roiMask = true(size(denoised));
        else
            numPixels = cellfun(@numel, cc.PixelIdxList);
            [~, idx] = max(numPixels);
            roiMask = false(size(denoised));
            roiMask(cc.PixelIdxList{idx}) = true;
        end
        roiMask = imdilate(roiMask, strel("disk", 7));

        roiPixels = enhanced(roiMask);
        if isempty(roiPixels)
            fillValue = mean(enhanced(:));
        else
            fillValue = mean(roiPixels);
        end
        masked = enhanced;
        masked(~roiMask) = uint8(fillValue);

        imwrite(masked, dstImagePath);

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
fprintf("Enhanced V4 ROI-segmented DIE_INK dataset saved to:\n%s\n", dstRoot);
fprintf("YOLO data file saved to:\n%s\n", dataYamlPath);
