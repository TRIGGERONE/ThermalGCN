import os
import glob
import numpy as np
from numpy import genfromtxt
import random
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Obtaining current dir/ path assignment/ Dir creation
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(BASE_DIR, 'data')
newdata_dir = os.path.join(BASE_DIR, 'newdata')

if not os.path.exists(newdata_dir):
    os.makedirs(newdata_dir)

# Processing single layout-power combination
def process_file(layout_power_tuple, data_dir, newdata_dir):
    layout, p = layout_power_tuple
    
    power_file = os.path.join(data_dir, f"Power_{layout}_{p}.csv")
    temp_file = os.path.join(data_dir, f"Temperature_{layout}_{p}.csv")
    edge_file = os.path.join(data_dir, f"Edge_{layout}_{p}.csv")
    
    # File check
    if not (os.path.exists(power_file) and os.path.exists(temp_file) and os.path.exists(edge_file)):
        print(f"Skipping incomplete set for layout {layout} power {p}")
        return None
    
    try:
        node_feats = genfromtxt(power_file, delimiter=',')
        node_labels = genfromtxt(temp_file, delimiter=',')
        edge_feats = genfromtxt(edge_file, delimiter=',')
        
        # Array check
        if node_feats.ndim < 2 or node_labels.ndim < 2 or edge_feats.ndim < 2:
            print(f"Skipping layout {layout} power {p} due to invalid data dimensions")
            return None
    except Exception as e:
        print(f"Error reading files for layout {layout} power {p}: {e}")
        return None
    
    # Calculation of power sum
    Power_sum = 0
    for m in range(node_feats.shape[0]):
        if m > 12291:
            node_feats[m][1] = 0.1
        Power_sum += node_feats[m][1]
    
    # Processing edge features
    for m in range(edge_feats.shape[0]):
        if edge_feats[m][0] in range(12288, 12300) and edge_feats[m][1] in range(12288, 12300):
            edge_feats[m][2] = 0.1
    
    # Calculation of min/max values 
    file_stats = {
        'Power_max': np.amax(node_feats[:, 1]),
        'Power_min': np.amin(node_feats[:, 1]),
        'Temperature_max': np.amax(node_labels[:, 1]),
        'Temperature_min': np.amin(node_labels[:, 1]),
        'Conductance_max': np.amax(edge_feats[:, 2]),
        'Conductance_min': np.amin(edge_feats[:, 2]),
        'tPower': Power_sum
    }
    
    # Processing 
    # Power data
    new_power_file = os.path.join(newdata_dir, f"Power_{layout}_{p}.csv")
    with open(new_power_file, 'w') as Power:
        for m in range(node_feats.shape[0]):
            if m > 12291:
                node_feats[m][1] = 0.1
            Power.write(f"{m},{node_feats[m][1]}\n")
    
    # Total power
    new_total_power_file = os.path.join(newdata_dir, f"totalPower_{layout}_{p}.csv")
    with open(new_total_power_file, 'w') as totalPower:
        for m in range(node_feats.shape[0]):
            totalPower.write(f"{Power_sum}\n")
    
    # Temperature data
    new_temp_file = os.path.join(newdata_dir, f"Temperature_{layout}_{p}.csv")
    with open(new_temp_file, 'w') as Temperature:
        for m in range(node_labels.shape[0]):
            Temperature.write(f"{m},{node_labels[m][1]}\n")
    
    # Edge data
    new_edge_file = os.path.join(newdata_dir, f"Edge_{layout}_{p}.csv")
    with open(new_edge_file, 'w') as Edge:
        for m in range(edge_feats.shape[0]):
            if edge_feats[m][0] in range(12288, 12300) and edge_feats[m][1] in range(12288, 12300):
                edge_feats[m][2] = 0.1
            Edge.write(f"{int(edge_feats[m][0])},{int(edge_feats[m][1])},{edge_feats[m][2]}\n")
    
    return file_stats


power_files = glob.glob(os.path.join(data_dir, "Power_*.csv"))

# Dictionary mapping: layout index to power indices
layout_dict = {}
for filepath in power_files:
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) != 3:
        continue
    
    try:
        layout_index = int(parts[1])
        power_index = int(parts[2].split('.')[0])
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        continue
        
    if layout_index not in layout_dict:
        layout_dict[layout_index] = []
    layout_dict[layout_index].append(power_index)

# Sorting indices 
available_layouts = sorted(layout_dict.keys())
for layout in available_layouts:
    layout_dict[layout].sort()

if not available_layouts:
    raise FileNotFoundError("No Power files found in the data directory.")

print(f"Found {len(available_layouts)} layouts with data")

# Initialization of global min/max values from first file
first_layout = available_layouts[0]
first_power = layout_dict[first_layout][0]

initial_stats = process_file((first_layout, first_power), data_dir, newdata_dir)
if not initial_stats:
    raise RuntimeError("Failed to process initial file")

global_stats = {
    'tPower_max': initial_stats['tPower'],
    'tPower_min': initial_stats['tPower'],
    'Power_max': initial_stats['Power_max'],
    'Power_min': initial_stats['Power_min'],
    'Temperature_max': initial_stats['Temperature_max'],
    'Temperature_min': initial_stats['Temperature_min'],
    'Conductance_max': initial_stats['Conductance_max'],
    'Conductance_min': initial_stats['Conductance_min']
}

file_tuples = [(layout, p) for layout in available_layouts for p in layout_dict[layout]]
remaining_files = [(layout, p) for layout, p in file_tuples if not (layout == first_layout and p == first_power)]

with mp.Pool(processes=mp.cpu_count()) as pool:
    results = list(tqdm(
        pool.imap(
            partial(process_file, data_dir=data_dir, newdata_dir=newdata_dir),
            remaining_files
        ),
        total=len(remaining_files),
        desc="Processing files"
    ))

for file_stats in results:
    if file_stats:
        global_stats['tPower_max'] = max(global_stats['tPower_max'], file_stats['tPower'])
        global_stats['tPower_min'] = min(global_stats['tPower_min'], file_stats['tPower'])
        global_stats['Power_max'] = max(global_stats['Power_max'], file_stats['Power_max'])
        global_stats['Power_min'] = min(global_stats['Power_min'], file_stats['Power_min'])
        global_stats['Temperature_max'] = max(global_stats['Temperature_max'], file_stats['Temperature_max'])
        global_stats['Temperature_min'] = min(global_stats['Temperature_min'], file_stats['Temperature_min'])
        global_stats['Conductance_max'] = max(global_stats['Conductance_max'], file_stats['Conductance_max'])
        global_stats['Conductance_min'] = min(global_stats['Conductance_min'], file_stats['Conductance_min'])

# Saving min/max values
maxmin_file1 = os.path.join(data_dir, "MaxMinValues.csv")
with open(maxmin_file1, 'w') as MaxMinFile:
    MaxMinFile.write("tPower_max, tPower_min, Power_max, Power_min, Temperature_max, Temperature_min, Conductance_max, Conductance_min\n")
    MaxMinFile.write(f"{global_stats['tPower_max']},{global_stats['tPower_min']},{global_stats['Power_max']},{global_stats['Power_min']},{global_stats['Temperature_max']},{global_stats['Temperature_min']},{global_stats['Conductance_max']},{global_stats['Conductance_min']}\n")

maxmin_file2 = os.path.join(newdata_dir, "MaxMinValues.csv")
with open(maxmin_file2, 'w') as MaxMinFile:
    MaxMinFile.write("tPower_max, tPower_min, Power_max, Power_min, Temperature_max, Temperature_min, Conductance_max, Conductance_min\n")
    MaxMinFile.write(f"{global_stats['tPower_max']},{global_stats['tPower_min']},{global_stats['Power_max']},{global_stats['Power_min']},{global_stats['Temperature_max']},{global_stats['Temperature_min']},{global_stats['Conductance_max']},{global_stats['Conductance_min']}\n")

# Train/test split 
all_layouts = available_layouts
num_layouts = len(all_layouts)
num_train = int(num_layouts * 0.85)
num_test = num_layouts - num_train

# Shuffling layouts before splitting 
random.shuffle(all_layouts)
train_layouts = all_layouts[:num_train]
test_layouts = all_layouts[num_train:]

train_test_file1 = os.path.join(newdata_dir, "test_data.csv")
train_test_file2 = os.path.join(newdata_dir, "train_data.csv")

with open(train_test_file1, 'w') as testFile, open(train_test_file2, 'w') as trainFile:
    # Test set
    for layout in test_layouts:
        for p in layout_dict[layout]:
            testFile.write(f"{layout}_{p}\n")
    
    # Train set
    for layout in train_layouts:
        for p in layout_dict[layout]:
            trainFile.write(f"{layout}_{p}\n")

# Counting samples
total_samples = sum(len(p_list) for p_list in layout_dict.values())
train_samples = sum(len(layout_dict[layout]) for layout in train_layouts)
test_samples = sum(len(layout_dict[layout]) for layout in test_layouts)

print("Preprocessing complete for synthetic dataset!")
print("Total samples detected:", total_samples)
print("Train samples:", train_samples)
print("Test samples:", test_samples)



