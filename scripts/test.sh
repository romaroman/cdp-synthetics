#!/bin/bash

devices=("Epson" "iP12w" "iPXS" "iP14w" "iP14uw" "iP15w" "iP15uw")
printers=("55" "76")
gpu="auto"

for device in "${devices[@]}"; do
	for printer in "${printers[@]}"; do
		python main.py --config-name general dataset.device="$device" dataset.printer="$printer" train.gpu="$gpu" prefix=experiment mode=train
		done
done