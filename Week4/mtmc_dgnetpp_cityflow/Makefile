PYTHON ?= python
DATA_ROOT ?= ../AI_CITY_CHALLENGE_2022_TRAIN
CONFIG ?= configs/default.yaml
TRAIN_SEQS ?= S01 S04
TEST_SEQ ?= S03
REID_DIR ?= outputs/reid_data
MODEL_DIR ?= outputs/reid_model
TRACK_FILE ?= outputs/track1.txt
FILTERED_TRACK_FILE ?= outputs/track1_filtered.txt
EVAL_DIR ?= ../AI_CITY_CHALLENGE_2022_TRAIN/eval
GT_S03 ?= $(EVAL_DIR)/ground_truth_s03.txt

.PHONY: help prepare train run eval-s03 postfilter clean-cache

help:
	@echo "Targets:"
	@echo "  make prepare      Extract ReID crops from S01/S04"
	@echo "  make train        Train DG-style ReID model"
	@echo "  make run          Run MTMC on S03"
	@echo "  make eval-s03     Evaluate TRACK_FILE on S03 ground truth"
	@echo "  make postfilter   Post-filter TRACK_FILE into FILTERED_TRACK_FILE"
	@echo "  make clean-cache  Remove Python cache files"

prepare:
	$(PYTHON) tools/prepare_reid_crops.py --data-root $(DATA_ROOT) --sequences $(TRAIN_SEQS) --output-dir $(REID_DIR)

train:
	$(PYTHON) tools/train_reid.py --config $(CONFIG) --train-index $(REID_DIR)/train_index.csv --output-dir $(MODEL_DIR)

run:
	$(PYTHON) tools/run_mtmc.py --config $(CONFIG) --data-root $(DATA_ROOT) --sequence $(TEST_SEQ) --checkpoint $(MODEL_DIR)/best.pt --output-file $(TRACK_FILE)

eval-s03:
	awk '$$1>=10 && $$1<=15' $(EVAL_DIR)/ground_truth_train.txt > $(GT_S03)
	cd $(EVAL_DIR) && $(PYTHON) eval.py ground_truth_s03.txt ../../mtmc_dgnetpp_cityflow/$(TRACK_FILE) --dstype train

postfilter:
	$(PYTHON) tools/postfilter_track1.py --input $(TRACK_FILE) --output $(FILTERED_TRACK_FILE) --min-global-cameras 3 --min-global-rows 150 --max-global-ids 12

clean-cache:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
