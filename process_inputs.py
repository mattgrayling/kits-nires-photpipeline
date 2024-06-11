import os
import shutil

data_loc = '/Users/matt/Downloads/KOA_15997/NIRES/raw/sci'

for folder in os.listdir(data_loc):
    if folder == '.DS_Store':
        continue
    date = folder.replace('-', '')
    if os.path.exists(f'data/{date}'):
        continue
    os.mkdir(f'data/{date}')
    for file in os.listdir(f'{data_loc}/{folder}'):
        if 'jpg' in file:
            continue
        shutil.copy(f'{data_loc}/{folder}/{file}', f'data/{date}')
