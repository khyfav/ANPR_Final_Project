# ISM 6642 Final Project — ANPR Prototype

This starter repo implements a scoped Automated Number Plate Reader (ANPR) prototype for Meridian Access Systems.

## Scope
- Assumes the input image is already cropped to the license plate.
- Builds the character classifier from EMNIST using PyTorch.
- Segments characters using OpenCV connected components/contours.
- Normalizes inference crops to the same 28x28 format used for training.
- Reconstructs plate text and calculates a plate-level confidence score.
- Routes a read to AUTO-ACCEPT, HUMAN-REVIEW, or MANUAL-ENTRY using configurable thresholds.
- Includes synthetic plate generation for known-ground-truth testing.

## Not in scope
- Vehicle detection
- Plate localization from full-scene images
- Tesseract/EasyOCR/PaddleOCR/cloud OCR as the classifier
- Production deployment

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the EMNIST model
```bash
python src/train_emnist.py --epochs 5 --batch-size 128
```
The first run downloads EMNIST via torchvision. The trained model is saved to `models/emnist_cnn.pt`.

## 2. Generate synthetic test plates
```bash
python src/generate_plates.py --count 100 --out data/synthetic
```
A `labels.csv` file is created with the exact ground-truth plate text.

## 3. Run evaluation
```bash
python src/evaluate_pipeline.py --image-dir data/synthetic --labels data/synthetic/labels.csv --model models/emnist_cnn.pt
```
The script reports segmentation success, character accuracy, plate accuracy, confidence, and common confusion pairs.

## 4. Launch the demo
```bash
streamlit run src/app.py
```
Upload an unseen cropped plate image. The app shows segmentation boxes, predicted characters, confidence, and routing decision.

## Recommended team split
- Data: EMNIST loader, class mapping, synthetic test data
- Model: CNN training, validation, error analysis
- Pipeline: segmentation, normalization, integration
- Demo: Streamlit interface and backup recording
- Business: cost framing, confidence policy, recommendation and roadmap

For 3–4 person teams, combine Business with another role as required by the assignment.

## Metrics to report
1. Character-level accuracy
2. Plate-level exact-match accuracy
3. Segmentation success rate
4. Accuracy by image condition
5. Confusion pairs / failure modes
6. Coverage by confidence threshold

Do not claim results until they are measured and reproducible.
