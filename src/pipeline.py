from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import EMNISTCNN


# --------------------------------------------------
# Character mappings
# --------------------------------------------------

CLASS_NAMES_36 = list(
    '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
)

CLASS_NAMES_62 = list(
    '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
)


# --------------------------------------------------
# Character segmentation
# --------------------------------------------------

def segment_characters(image_bgr):
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr.copy()
    )

    h, w = gray.shape[:2]

    # Focus only on the main license-number band.
    y1 = int(h * 0.25)
    y2 = int(h * 0.82)

    x1 = int(w * 0.06)
    x2 = int(w * 0.94)

    roi = gray[
        y1:y2,
        x1:x2
    ]

    # Convert characters to white foreground on black background.
    _, bw = cv2.threshold(
        roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_h, roi_w = roi.shape[:2]

    boxes = []

    # --------------------------------------------------
    # Initial contour filtering
    # --------------------------------------------------

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        height_ratio = ch / roi_h
        width_ratio = cw / roi_w
        aspect_ratio = cw / float(ch)

        # Character should occupy a reasonable amount
        # of the plate height.
        if height_ratio < 0.45 or height_ratio > 0.98:
            continue

        # Remove tiny artifacts and very wide graphics.
        if width_ratio < 0.02 or width_ratio > 0.22:
            continue

        if aspect_ratio < 0.10 or aspect_ratio > 1.25:
            continue

        # Convert ROI coordinates back to full-image coordinates.
        boxes.append(
            (
                x + x1,
                y + y1,
                cw,
                ch
            )
        )

    # --------------------------------------------------
    # Keep boxes belonging to the main character row
    # --------------------------------------------------

    if boxes:

        centers_y = [
            y + (ch / 2.0)
            for x, y, cw, ch in boxes
        ]

        median_center_y = np.median(
            centers_y
        )

        heights = [
            ch
            for x, y, cw, ch in boxes
        ]

        median_height = np.median(
            heights
        )

        filtered_boxes = []

        for box in boxes:
            x, y, cw, ch = box

            center_y = (
                y + (ch / 2.0)
            )

            vertical_ok = (
                abs(
                    center_y - median_center_y
                )
                <
                (
                    median_height * 0.35
                )
            )

            height_ok = (
                median_height * 0.70
                <= ch
                <= median_height * 1.30
            )

            if vertical_ok and height_ok:
                filtered_boxes.append(
                    box
                )

        boxes = filtered_boxes

    # Sort left-to-right.
    boxes.sort(
        key=lambda b: b[0]
    )

    # --------------------------------------------------
    # Extract character crops
    # --------------------------------------------------

    crops = []

    for x, y, cw, ch in boxes:
        crop = gray[
            y:y + ch,
            x:x + cw
        ]

        crops.append(
            crop
        )

    return crops, boxes, bw


# --------------------------------------------------
# Normalize character crop to EMNIST-style 28x28
# --------------------------------------------------

def normalize_crop(crop):

    _, mask = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ys, xs = np.where(
        mask > 0
    )

    # Return blank image if nothing is found.
    if len(xs) == 0:
        return np.zeros(
            (28, 28),
            dtype=np.uint8
        )

    roi = mask[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    h, w = roi.shape

    # Scale character into a 20x20 area
    # while preserving aspect ratio.
    scale = min(
        20 / max(w, 1),
        20 / max(h, 1)
    )

    nw = max(
        1,
        int(w * scale)
    )

    nh = max(
        1,
        int(h * scale)
    )

    roi = cv2.resize(
        roi,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    # Center character in 28x28 field.
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x0 = (
        28 - nw
    ) // 2

    y0 = (
        28 - nh
    ) // 2

    canvas[
        y0:y0 + nh,
        x0:x0 + nw
    ] = roi

    return canvas


# --------------------------------------------------
# Load either 36-class or 62-class CNN
# --------------------------------------------------

def load_model(model_path):

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    num_classes = checkpoint.get(
        'num_classes',
        62
    )

    model = EMNISTCNN(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint['state_dict']
    )

    model.eval()

    # Store class mapping directly on model.
    if num_classes == 36:
        model.class_names = CLASS_NAMES_36

    elif num_classes == 62:
        model.class_names = CLASS_NAMES_62

    else:
        raise ValueError(
            f"Unsupported number of classes: {num_classes}"
        )

    model.num_classes = num_classes

    print(
        f"Loaded {num_classes}-class model "
        f"on {device}"
    )

    return model, device


# --------------------------------------------------
# Predict one character
# --------------------------------------------------

def predict_char(
    model,
    device,
    normalized
):

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        ),
    ])

    image = Image.fromarray(
        normalized
    )

    x = tfm(
        image
    ).unsqueeze(0).to(
        device
    )

    with torch.no_grad():

        logits = model(
            x
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        confidence, index = probs.max(
            0
        )

    idx = index.item()

    character = model.class_names[
        idx
    ]

    return (
        character,
        confidence.item()
    )


# --------------------------------------------------
# Business routing decision
# --------------------------------------------------

def route_decision(
    plate_confidence,
    accept_threshold=0.95,
    review_threshold=0.80
):

    if plate_confidence >= accept_threshold:
        return 'AUTO-ACCEPT'

    if plate_confidence >= review_threshold:
        return 'HUMAN-REVIEW'

    return 'MANUAL-ENTRY'


# --------------------------------------------------
# End-to-end plate reading
# --------------------------------------------------

def read_plate(
    image_path,
    model,
    device
):

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        raise ValueError(
            f'Could not read image: {image_path}'
        )

    crops, boxes, bw = segment_characters(
        img
    )

    predictions = []
    confidences = []

    for crop in crops:

        normalized = normalize_crop(
            crop
        )

        character, confidence = predict_char(
            model,
            device,
            normalized
        )

        predictions.append(
            character
        )

        confidences.append(
            confidence
        )

    text = ''.join(
        predictions
    )

    # Conservative confidence:
    # weakest predicted character determines
    # overall plate confidence.
    plate_confidence = (
        min(confidences)
        if confidences
        else 0.0
    )

    return {
        'text': text,
        'character_confidences': confidences,
        'plate_confidence': plate_confidence,
        'decision': route_decision(
            plate_confidence
        ),
        'boxes': boxes,
        'binary': bw,
        'image': img,
    }
