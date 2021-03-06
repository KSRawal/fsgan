#!/usr/bin/env bash
conda activate FSGAN

export INPUT_VIDEO='./data/KSR animated.mp4'
export TARGET_VIDEO='./data/TomOblivion.mp4'

python swap.py \
	"$INPUT_VIDEO" \
	-t "$TARGET_VIDEO" \
	-o . \
	--finetune \
	--finetune_save \
	--seg_remove_mouth
