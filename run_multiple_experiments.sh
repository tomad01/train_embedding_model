#!/bin/bash

# Define the list of parameters and Python scripts to run
parameters=("./comfig/config.yml")
scripts=("baseline_experiment.py")

# Define the log file
log_file="log.txt"

# Clear the log file if it exists
> "$log_file"

# Iterate over each script and parameter, and run the script with the parameter
for script in "${scripts[@]}"; do
    for param in "${parameters[@]}"; do
        echo "Running $script with parameter $param..."
        echo "Running $script with parameter $param..." >> "$log_file"
        python "$script" "$param"
        
        # Check if the script executed successfully
        if [ $? -ne 0 ]; then
            error_message="Error: $script failed to run with parameter $param."
            echo "$error_message"
            echo "$error_message" >> "$log_file"
        fi
    done
done

echo "All scripts executed."

