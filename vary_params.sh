#!/bin/bash
epochs=100000
alpha=0.01

for hidden in 32 64 128; do
    for features in 2 4 6; do
        echo "Running train.py with cross interaction: num-hidden=$hidden, num-features=$features"
        
        # Unregularized
        python train.py --num-hidden "$hidden" --num-cues "$features" \
            --save-path "trained_models/pretrained_cross_${hidden}_${features}_" \
            --epochs "$epochs"
        
        # Regularized
        python train.py --alpha "$alpha" --num-cues "$hidden" --num-features "$features" \
            --save-path "trained_models/alpha_cross_${hidden}_${features}_" \
            --load-path "trained_models/pretrained_cross_${hidden}_${features}_0.pth" \
            --epochs "$epochs"
    done
done

echo "All runs completed."
