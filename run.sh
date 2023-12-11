#!/bin/bash

#Define Variables
PYTHON_EXEC="python3"
DATA_SCRIPT="data.py"
MODEL_SCRIPT="model.py"
TRAIN_SCRIPT="train.py"
CONFIG_FILE="config.json"


$PYTHON_EXEC $DATA_SCRIPT --config $CONFIG_FILE
$PYTHON_EXEC $MODEL_SCRIPT --config $CONFIG_FILE
$PYTHON_EXEC $TRAIN_SCRIPT --config $CONFIG_FILE

echo "Training Completed"
