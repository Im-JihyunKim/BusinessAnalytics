import os
import argparse



def mypath(args: argparse):
    top_save_path = args.save_dir

    if not os.path.exists(top_save_path):
        os.makedirs(top_save_path)

    if len(os.listdir(top_save_path)) > 0:
        x = len(os.listdir(top_save_path))
        x = x + 1

    else:
        x = 1

    # save_path = f'./tas_results/Experiment_{x}'
    save_path = os.path.join(top_save_path, f'Experiment_{x}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path
