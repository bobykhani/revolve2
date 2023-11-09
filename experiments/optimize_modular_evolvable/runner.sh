#!/bin/bash

# Directory containing your Python files
DIRECTORY="."

# Name of your virtual environment directory
VENV=".venv"

# Iterate over each Python file in the directory starting with "optimize_" and ending with ".py"
for FILE in $DIRECTORY/optimize_*.py; do
    # Extract the filename for window naming purposes
    BASENAME=$(basename "$FILE" .py)
    
    # Create a new tmux window
    tmux new-window -n "$BASENAME"
    
    # Activate the virtual environment and run the Python file
    tmux send-keys "source $DIRECTORY/../../$VENV/bin/activate" C-m
    tmux send-keys "python $FILE" C-m
done

