import glob
import os




# For results of generate_synthetic.py


# Get the absolute path of the directory containing this script.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Count floorplan files (Synthetic_Chiplet_*.flp).
flp_pattern = os.path.join(BASE_DIR, "Synthetic_Chiplet_*.flp")
flp_files = glob.glob(flp_pattern)
print("Number of floorplan files generated:", len(flp_files))

# Count power trace files (Synthetic_Chiplet_*_Power*.ptrace).
ptrace_pattern = os.path.join(BASE_DIR, "Synthetic_Chiplet_*_Power*.ptrace")
ptrace_files = glob.glob(ptrace_pattern)
print("Number of power trace files generated:", len(ptrace_files))













# For results of run_synthetic.py

data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

power_files = glob.glob(os.path.join(data_dir, "Power_*.csv"))
temperature_files = glob.glob(os.path.join(data_dir, "Temperature_*.csv"))
edge_files = glob.glob(os.path.join(data_dir, "Edge_*.csv"))

print("Number of Power files:", len(power_files))
print("Number of Temperature files:", len(temperature_files))
print("Number of Edge files:", len(edge_files))





# Total number of layouts expected
num_layouts = 400

missing_files = []

for i in range(num_layouts):
    if i == 0:
        # For layout 0, we expect "Chiplet_Core.flp"
        expected_file = "Chiplet_Core.flp"
    else:
        expected_file = f"Synthetic_Chiplet_{i}.flp"
        
    if not os.path.exists(expected_file):
        missing_files.append(expected_file)
        print(f"Missing: {expected_file}")

if not missing_files:
    print("All expected floorplan files are present!")
else:
    print(f"\nTotal missing files: {len(missing_files)}")








# For results of data_prerpocess_synthetic.py


# Set the path to the newdata folder
newdata_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'newdata')

# Define file paths
train_file = os.path.join(newdata_dir, 'train_data.csv')
test_file  = os.path.join(newdata_dir, 'test_data.csv')

def count_lines(filepath):
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

train_count = count_lines(train_file)
test_count  = count_lines(test_file)
total_count = train_count + test_count

print("Train samples:", train_count)
print("Test samples: ", test_count)
print("Total samples:", total_count)
