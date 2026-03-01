.PHONY: help fetch crop empty synth prelabel labeler split train eval clean

help:
	@echo "ParkOpticon YOLO11s Bootstrap - Pipeline Commands"
	@echo ""
	@echo "make fetch       - Download Street View images"
	@echo "make crop       - Crop fetched images from bottom"
	@echo "make empty       - Detect empty frames (no vehicles)"
	@echo "make synth      - Generate synthetic vehicle edits"
	@echo "make prelabel   - Auto-generate bounding box labels"
	@echo "make labeler    - Start labeling UI"
	@echo "make split      - Create train/val/test split"
	@echo "make train      - Train YOLO11s detector"
	@echo "make eval       - Evaluate on test set"
	@echo "make clean      - Clean generated files"

fetch:
	python scripts/02_fetch_streetview.py

crop:
	python scripts/02b_crop_bottom.py

empty:
	python scripts/03_detect_empty_frames.py

synth:
	python scripts/04_synthesize_vehicle_edits.py

prelabel:
	python scripts/05_prelabel_boxes.py

labeler:
	cd labeler && python app.py

split:
	python scripts/06_split_dataset.py

train:
	python scripts/07_train_yolo.py

eval:
	python scripts/08_eval_report.py --weights runs/detect/parkopticon_vehicle_enforcement/weights/best.pt

clean:
	rm -rf data/images_original/*/
	rm -rf data/images_synth/*/
	rm -rf data/labels_autogen/*
	rm -rf data/labels_final/*
	rm -rf data/splits/*/
	rm -rf runs/detect/*
	rm -rf reports/*
	rm -f manifests/images.csv
	rm -f manifests/boxes_autogen.jsonl
	rm -f lists/empty_candidates.txt
