import cv2
import numpy as np
from scipy import ndimage
import torch

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "./weights/sam_hq_vit_l.pth"
model_type = "vit_l"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


def rescale_image(image, px=512, padding=0):
    height, width, _ = image.shape
    if [height, width].index(max([height, width])) == 0:
        factor = px / height
        height = px
        width = int(width * factor)
    else:
        factor = px / width
        width = px
        height = int(height * factor)

    image_resized = cv2.resize(
        image, dsize=(width, height), interpolation=cv2.INTER_LINEAR
    )

    # Create a larger canvas with the same number of channels as the input image
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_image = np.zeros(
        (padded_height, padded_width, image.shape[2]), dtype=np.uint8
    )

    # Calculate the position to place the resized image in the center
    x_offset = (padded_width - width) // 2
    y_offset = (padded_height - height) // 2

    # Place the resized image in the center of the padded canvas
    padded_image[
        y_offset : y_offset + height, x_offset : x_offset + width
    ] = image_resized

    return padded_image


def add_outline(image, stroke_size, outline_color):
    # Ensure the image has an alpha channel for transparency
    if image.shape[-1] != 4:
        raise ValueError("Input image must have an alpha channel (4 channels).")

    # Create a copy of the original image
    outlined_image = image.copy()

    # Create a mask for fully transparent parts of the image
    mask = (image[:, :, 3] == 0).astype(np.uint8)

    # Calculate the kernel size based on the desired stroke size
    kernel_size = int(stroke_size * 0.2) * 2 + 1  # Ensure it's an odd number
    if kernel_size < 1:
        kernel_size = 1

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion to round the outline
    outline = cv2.erode(mask, kernel, iterations=1)

    # Use the eroded mask to get the outline of the fully transparent parts
    outline = mask - outline

    # Apply Gaussian blur to smooth the outline
    outline = cv2.GaussianBlur(
        outline.astype(np.float32), (kernel_size, kernel_size), 0
    )

    # Threshold the blurred outline to make it binary
    _, outline = cv2.threshold(outline, 0.5, 1, cv2.THRESH_BINARY)

    # Apply the outline color only to the outline region
    for c in range(4):  # Loop through RGBA channels
        outlined_image[:, :, c] = (
            outlined_image[:, :, c] * (1 - outline) + outline_color[c] * outline
        )

    return outlined_image


def extract_bounding_box(image, bbox):
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        bounding_box_image = image[min_y : max_y + 1, min_x : max_x + 1]
        return bounding_box_image
    else:
        return None


def get_bbox_from_image(image):
    alpha_channel = image[
        :, :, 3
    ]  # Assuming alpha channel is the last channel (index 3)
    non_transparent_mask = alpha_channel > 0

    # Find the coordinates of non-transparent pixels
    non_transparent_indices = np.argwhere(non_transparent_mask)

    if non_transparent_indices.size > 0:
        x1, y1 = non_transparent_indices.min(axis=0)
        x2, y2 = non_transparent_indices.max(axis=0)

        # Bounding box coordinates
        x1, y1, x2, y2 = y1, x1, y2, x2  # Swap x and y for correct format
        return x1, y1, x2, y2
    return False


def keep_small_transparent_regions(mask, h_area_threshold=None, w_area_threshold=None):
    if h_area_threshold is None:
        h_area_threshold = mask.shape[0] * 0.05
    if w_area_threshold is None:
        w_area_threshold = mask.shape[1] * 0.05

    labeled, num_features = ndimage.label(mask == 0)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes[labeled]
    mask_filled = mask.copy()

    for label in range(1, num_features + 1):
        area = np.sum(labeled == label)
        if area <= h_area_threshold * w_area_threshold:
            h, w = np.where(labeled == label)
            mask_filled[h, w] = 1

    return mask_filled


def remove_small_nontransparent_regions(
    mask, h_area_threshold=None, w_area_threshold=None
):
    if h_area_threshold is None:
        h_area_threshold = mask.shape[0] * 0.05
    if w_area_threshold is None:
        w_area_threshold = mask.shape[1] * 0.05

    labeled, num_features = ndimage.label(mask == 1)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes[labeled]
    mask_removed = mask.copy()

    for label in range(1, num_features + 1):
        area = np.sum(labeled == label)
        if area <= h_area_threshold * w_area_threshold:
            h, w = np.where(labeled == label)
            mask_removed[h, w] = 0

    return mask_removed


def segment(input_path, boxes=None, points=None):
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    hq_token_only = True
    output_path = f"{input_path.split('.')[0]}.png"

    if points:
        input_point = np.array(points)
        input_label = np.ones(input_point.shape[0])
        input_box = None
        transformed_box = None
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
    elif boxes:
        input_box = torch.tensor(boxes, device=predictor.device)
        transformed_box = predictor.transform.apply_boxes_torch(
            input_box, image.shape[:2]
        )
        input_point, input_label = None, None
        masks, scores, logits = predictor.predict_torch(
            point_coords=input_point,
            point_labels=input_label,
            boxes=transformed_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        masks = masks.numpy().squeeze(1)

    image = cv2.imread(input_path)
    combined_mask = np.sum(masks, axis=0)
    mask = combined_mask.astype(np.uint8)
    mask = keep_small_transparent_regions(mask)
    mask = remove_small_nontransparent_regions(mask)
    alpha_channel = np.where(mask == 0, 0, 255).astype(np.uint8)
    image = cv2.merge((image, alpha_channel))

    return image


def make_sticker(input_path, output_path, boxes=None, points=None):
    image = segment(input_path, boxes, points)
    bbox = get_bbox_from_image(image)
    if isinstance(bbox, bool) and bbox is False:
        return False
    image = extract_bounding_box(image, bbox)
    image = rescale_image(image, padding=13)
    image = add_outline(image, 40, (255, 255, 255, 255))
    image = rescale_image(image, padding=0)

    cv2.imwrite(output_path, image)
    return True
