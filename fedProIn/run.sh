#!/bin/bash
num_repeats=${3:-3}
num_experiments=$( python -c "from modify_configs import experiments; print(len(experiments))" )

start_index=${1:-0}
end_index=${2:-40}

mkdir -p out

# Signal handler to restore backup on Ctrl+C
# Signal handler to restore backup on Ctrl+C
cleanup() {
    echo ""
    echo "Interrupted! Restoring backup..."
    python modify_configs.py --index -1
    exit 1
}

# Set up signal handler for SIGINT (Ctrl+C)
trap cleanup SIGINT

if (( start_index < 0 )); then
    echo "Invalid start index: $start_index"
    exit 1
fi

if (( start_index > end_index )); then
    echo "Start index $start_index is greater than end index $end_index"
    exit 1
fi

if (( end_index >= num_experiments )); then
    echo "Adjusted end index from $end_index to $((num_experiments - 1)) to match available experiments."
    end_index=$((num_experiments - 1))
fi

for ((exp=start_index; exp<=end_index; exp++))
do
    echo "=============================="
    echo " Running Experiment $exp "
    echo "=============================="
    python modify_configs.py --index $exp

    for ((i=1; i<=num_repeats; i++))
    do
        echo "  Run $i for experiment $exp..."
        start_time=$SECONDS
        nohup python main.py > "out/out_exp${exp}_run${i}.txt" 2>&1
        end_time=$SECONDS
        echo "Time taken for run $i: $((end_time - start_time)) seconds"
    done
    python modify_configs.py --index -1
done

echo "All experiments completed."