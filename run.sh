#!/usr/bin/env bash

gpu_id=0  # Set GPU ID.

# Run experiments on PACS dataset for the single labeled domain generalization (SLDG) task.
for data in 0 1 2 3; do  # Let target_data = 0(Ar), 1(Ca), 2(Ph), 3(Sk).
    python main.py --gpu_id ${gpu_id} --step 10 --d_name pacs --t_da_i ${data} > pacs-SLDG-data${data}.txt;
done


# Run experiments on Office-Home dataset for the single labeled domain generalization (SLDG) task.
for data in 0 1 2 3; do  # Let target_data = 0(Ar), 1(Cl), 2(Pr), 3(Rw).
    python main.py --gpu_id ${gpu_id} --step 15 --max_epoch 30 --d_name office-home --t_da_i ${data} > office-home-SLDG-data${data}.txt;
done


# Run experiments on PACS dataset for the conventional domain generalization (CDG) task.
for data in 0 1 2 3; do  # Let target_data = 0(Ar), 1(Ca), 2(Ph), 3(Sk).
    python main.py --gpu_id ${gpu_id} --step 10 --d_name pacs --t_da_i ${data} --pseudo 0 > pacs-DG-data${data}.txt;
done


# Run experiments on Office-Home dataset for the conventional domain generalization (CDG) task.
for data in 0 1 2 3; do  # Let target_data = 0(Ar), 1(Cl), 2(Pr), 3(Rw).
    python main.py --gpu_id ${gpu_id} --step 15 --max_epoch 30 --d_name office-home --t_da_i ${data} --pseudo 0 > office-home-DG-data${data}.txt;
done