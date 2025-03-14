# import os
# import time

# # Create required directories if they don't exist
# if not os.path.exists('./data'):
#     os.makedirs('./data')
# if not os.path.exists('./data'):
#     os.makedirs('./data')

# # Process all synthetic layouts
# for i in range(400):
#     # Get chip size from layout_params
#     chip_size = 12  # Default size
#     with open('._config/layout_params.csv', 'r') as f:
#         lines = f.readlines()
#         for line in lines[1:]:  # Skip header
#             parts = line.strip().split(',')
#             if int(parts[0]) == i:
#                 chip_size = int(parts[2])
#                 break
    
#     # Calculate grid resolution based on chip size
#     grid_resolution = int(64 * (chip_size / 12))
    
#     # Special handling for the first file which is already named Chiplet_Core.flp
#     if i == 0:
#         synthetic_file = "Chiplet_Core.flp"
#         # No need to rename it since it's already correctly named
#     else:
#         synthetic_file = f"Synthetic_Chiplet_{i}.flp"
#         os.rename(synthetic_file, "Chiplet_Core.flp")
    
#     # Process each power variation
#     for j in range(20):
#         # Run HotSpot simulation with dynamic grid resolution
#         cmd = f"../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p Synthetic_Chiplet_{i}_Power{j}.ptrace -steady_file Chiplet.steady -model_type grid -grid_rows {grid_resolution} -grid_cols {grid_resolution} -grid_steady_file Chiplet.grid.steady"
        
#         tmr_start = time.time()
#         os.system(cmd)
#         tmr_end = time.time()
#         print(f"Layout {i}, Power {j}: {tmr_end - tmr_start}s")
        
#         # Check if output files exist before renaming
#         if os.path.exists("./data/Edge.csv"):
#             edge_file = f"./data/Edge_{i}_{j}.csv"
#             os.rename("./data/Edge.csv", edge_file)
        
#         if os.path.exists("./data/Temperature.csv"):
#             temp_file = f"./data/Temperature_{i}_{j}.csv"
#             os.rename("./data/Temperature.csv", temp_file)
        
#         if os.path.exists("./data/Power.csv"):
#             power_file = f"./data/Power_{i}_{j}.csv"
#             os.rename("./data/Power.csv", power_file)
    
#     # Restore original filename only if we renamed it
#     if i != 0:
#         os.rename("Chiplet_Core.flp", synthetic_file)
    
#     print(f"Completed layout {i}")

# print("Thermal simulation complete for all synthetic layouts!")



# ========================================================================================================================================================================








#WORKING CODE







# import os
# import time

# # Create required directories if they don't exist
# if not os.path.exists('./data'):
#     os.makedirs('./data')
# if not os.path.exists('./data'):
#     os.makedirs('./data')

# # Process all synthetic layouts
# for i in range(400):
#     # Get chip size from layout_params
#     chip_size = 12  # Default size
#     try:
#         with open('./layout_params.csv', 'r') as f:
#             lines = f.readlines()
#             for line in lines[1:]:  # Skip header
#                 parts = line.strip().split(',')
#                 if int(parts[0]) == i:
#                     chip_size = int(parts[2])
#                     break
#     except FileNotFoundError:
#         print(f"Warning: layout_params.csv not found, using default chip size of 12mm")
    
#     # Calculate grid resolution based on chip size with bounds
#     grid_resolution = min(max(int(64 * (chip_size / 12)), 32), 512)

    

#     synthetic_file = f"Synthetic_Chiplet_{i}.flp"
#     if os.path.exists(synthetic_file):
#         os.rename(synthetic_file, "Chiplet_Core.flp")
#     else:
#         print(f"Warning: {synthetic_file} not found, skipping layout {i}")
#         continue
    
#     # Process each power variation
#     for j in range(20):
#         power_file_name = f"Synthetic_Chiplet_{i}_Power{j}.ptrace"
#         if not os.path.exists(power_file_name):
#             print(f"Warning: {power_file_name} not found, skipping power variation {j}")
#             continue
            
#         # Run HotSpot simulation with dynamic grid resolution
#         cmd = f"../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p {power_file_name} -steady_file Chiplet.steady -model_type grid -grid_rows {grid_resolution} -grid_cols {grid_resolution} -grid_steady_file Chiplet.grid.steady"
        
#         tmr_start = time.time()
#         os.system(cmd)
#         tmr_end = time.time()
#         print(f"Layout {i}, Power {j}: {tmr_end - tmr_start}s")
        
#         # Check if output files exist before renaming
#         if os.path.exists("./data/Edge.csv"):
#             edge_file = f"./data/Edge_{i}_{j}.csv"
#             os.rename("./data/Edge.csv", edge_file)
#         else:
#             print(f"Warning: Edge.csv not generated for layout {i}, power {j}")
        
#         if os.path.exists("./data/Temperature.csv"):
#             temp_file = f"./data/Temperature_{i}_{j}.csv"
#             os.rename("./data/Temperature.csv", temp_file)
#         else:
#             print(f"Warning: Temperature.csv not generated for layout {i}, power {j}")
        
#         if os.path.exists("./data/Power.csv"):
#             power_file = f"./data/Power_{i}_{j}.csv"
#             os.rename("./data/Power.csv", power_file)
#         else:
#             print(f"Warning: Power.csv not generated for layout {i}, power {j}")
    
#     # Restore original filename only if we renamed it
#     if i != 0 and os.path.exists("Chiplet_Core.flp"):
#         os.rename("Chiplet_Core.flp", synthetic_file)
    
#     print(f"Completed layout {i}")

# print("Thermal simulation complete for all synthetic layouts!")



# ========================================================================================================================================================================



















#Can use above code instead



# import os
# import time

# # Create required directories if they don't exist
# if not os.path.exists('./data'):
#     os.makedirs('./data')
# if not os.path.exists('./data'):
#     os.makedirs('./data')

# # Process all synthetic layouts
# for i in range(400):
#     # Get chip size from layout_params
#     chip_size = 12  # Default size
#     try:
#         with open('./layout_params.csv', 'r') as f:
#             lines = f.readlines()
#             for line in lines[1:]:  # Skip header
#                 parts = line.strip().split(',')
#                 if int(parts[0]) == i:
#                     chip_size = int(parts[2])
#                     break
#     except FileNotFoundError:
#         print(f"Warning: layout_params.csv not found, using default chip size of 12mm")
    
#     # Calculate grid resolution based on chip size with bounds
#     grid_resolution = min(max(int(64 * (chip_size / 12)), 32), 512)

#     synthetic_file = f"Synthetic_Chiplet_{i}.flp"
#     if os.path.exists(synthetic_file):
#         os.rename(synthetic_file, "Chiplet_Core.flp")
#     else:
#         print(f"Warning: {synthetic_file} not found, skipping layout {i}")
#         continue
    
#     # Process each power variation
#     for j in range(20):
#         power_file_name = f"Synthetic_Chiplet_{i}_Power{j}.ptrace"
#         if not os.path.exists(power_file_name):
#             print(f"Warning: {power_file_name} not found, skipping power variation {j}")
#             continue
        
#         # Try different grid resolutions if the first one fails
#         grid_resolutions = [grid_resolution, 64, 128, 256]
#         success = False
        
#         for res in grid_resolutions:
#             # Run HotSpot simulation with current grid resolution
#             cmd = f"../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p {power_file_name} -steady_file Chiplet.steady -model_type grid -grid_rows {res} -grid_cols {res} -grid_steady_file Chiplet.grid.steady"
            
#             tmr_start = time.time()
#             result = os.system(cmd)
#             tmr_end = time.time()
            
#             print(f"Layout {i}, Power {j}, Grid {res}: {tmr_end - tmr_start}s")
            
#             # Check if command succeeded
#             if result == 0 and os.path.exists("./data/Edge.csv"):
#                 success = True
#                 break
        
#         if not success:
#             print(f"Failed to simulate layout {i}, power {j} with all grid resolutions")
#             continue
            
#         # Check if output files exist before renaming
#         if os.path.exists("./data/Edge.csv"):
#             edge_file = f"./data/Edge_{i}_{j}.csv"
#             os.rename("./data/Edge.csv", edge_file)
#         else:
#             print(f"Warning: Edge.csv not generated for layout {i}, power {j}")
        
#         if os.path.exists("./data/Temperature.csv"):
#             temp_file = f"./data/Temperature_{i}_{j}.csv"
#             os.rename("./data/Temperature.csv", temp_file)
#         else:
#             print(f"Warning: Temperature.csv not generated for layout {i}, power {j}")
        
#         if os.path.exists("./data/Power.csv"):
#             power_file = f"./data/Power_{i}_{j}.csv"
#             os.rename("./data/Power.csv", power_file)
#         else:
#             print(f"Warning: Power.csv not generated for layout {i}, power {j}")
    
#     # Restore original filename only if we renamed it
#     if i != 0 and os.path.exists("Chiplet_Core.flp"):
#         os.rename("Chiplet_Core.flp", synthetic_file)
    
#     print(f"Completed layout {i}")

# print("Thermal simulation complete for all synthetic layouts!")




# ========================================================================================================================================================================





#  WORKING CODE WITH 3 TIMES RETRY FOR PROBLEMENATIC LAYOUTS






# import os
# import time
# import glob

# # Create required directories if they don't exist
# if not os.path.exists('./data'):
#     os.makedirs('./data')
# if not os.path.exists('./data'):
#     os.makedirs('./data')

# # Check which combinations are already processed
# existing_files = set()
# for file in glob.glob("./data/Power_*_*.csv"):
#     basename = os.path.basename(file)  # e.g., "Power_123_7.csv"
#     parts = basename.split('_')
#     if len(parts) == 3:
#         layout_id = int(parts[1])
#         power_id = int(parts[2].split('.')[0])
#         existing_files.add((layout_id, power_id))

# print(f"Found {len(existing_files)} existing processed files")

# # Process all synthetic layouts
# for i in range(400):
#     # Get chip size from layout_params
#     chip_size = 12  # Default size
#     try:
#         with open('./layout_params.csv', 'r') as f:
#             lines = f.readlines()
#             for line in lines[1:]:  # Skip header
#                 parts = line.strip().split(',')
#                 if int(parts[0]) == i:
#                     chip_size = int(parts[2])
#                     break
#     except FileNotFoundError:
#         print(f"Warning: layout_params.csv not found, using default chip size of 12mm")

#     # Calculate grid resolution based on chip size with bounds
#     grid_resolution = min(max(int(64 * (chip_size / 12)), 32), 512)
    
#     synthetic_file = f"Synthetic_Chiplet_{i}.flp"
#     if not os.path.exists(synthetic_file):
#         print(f"Warning: {synthetic_file} not found, skipping layout {i}")
#         continue
        
#     # Process each power variation
#     for j in range(20):
#         # Skip if this combination is already processed
#         if (i, j) in existing_files:
#             print(f"Skipping existing file for Layout {i}, Power {j}")
#             continue
            
#         power_file_name = f"Synthetic_Chiplet_{i}_Power{j}.ptrace"
#         if not os.path.exists(power_file_name):
#             print(f"Warning: {power_file_name} not found, skipping power variation {j}")
#             continue

#         # Rename the floorplan file to Chiplet_Core.flp for HotSpot
#         os.rename(synthetic_file, "Chiplet_Core.flp")
        
#         # Try different grid resolutions if the first one fails
#         grid_resolutions = [grid_resolution, 64, 128, 256, 512]
#         success = False
        
#         for res in grid_resolutions:
#             # Run HotSpot simulation with current grid resolution
#             cmd = f"../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p {power_file_name} -steady_file Chiplet.steady -model_type grid -grid_rows {res} -grid_cols {res} -grid_steady_file Chiplet.grid.steady"
            
#             tmr_start = time.time()
#             result = os.system(cmd)
#             tmr_end = time.time()
            
#             print(f"Layout {i}, Power {j}, Grid {res}: {tmr_end - tmr_start}s")
            
#             # Check if command succeeded and output files exist
#             if result == 0 and os.path.exists("./data/Edge.csv"):
#                 success = True
#                 break
        
#         if not success:
#             print(f"Failed to simulate layout {i}, power {j} with all grid resolutions")
#             # Restore original filename
#             os.rename("Chiplet_Core.flp", synthetic_file)
#             continue
            
#         # Check if output files exist before renaming
#         if os.path.exists("./data/Edge.csv"):
#             edge_file = f"./data/Edge_{i}_{j}.csv"
#             os.rename("./data/Edge.csv", edge_file)
#         else:
#             print(f"Warning: Edge.csv not generated for layout {i}, power {j}")
            
#         if os.path.exists("./data/Temperature.csv"):
#             temp_file = f"./data/Temperature_{i}_{j}.csv"
#             os.rename("./data/Temperature.csv", temp_file)
#         else:
#             print(f"Warning: Temperature.csv not generated for layout {i}, power {j}")
            
#         if os.path.exists("./data/Power.csv"):
#             power_file = f"./data/Power_{i}_{j}.csv"
#             os.rename("./data/Power.csv", power_file)
#         else:
#             print(f"Warning: Power.csv not generated for layout {i}, power {j}")
            
#         # Restore original filename
#         os.rename("Chiplet_Core.flp", synthetic_file)
        
#     print(f"Completed layout {i}")
    
# print("Thermal simulation complete for all synthetic layouts!")





# ========================================================================================================================================================================








import os
import time
import glob

# Create required directories if they don't exist
if not os.path.exists('./data'):
    os.makedirs('./data')
# if not os.path.exists('./data'):
#     os.makedirs('./data')

# Check which combinations are already processed
existing_files = set()
for file in glob.glob("./data/Power_*_*.csv"):
    basename = os.path.basename(file)
    parts = basename.split('_')
    if len(parts) == 3:
        layout_id = int(parts[1])
        power_id = int(parts[2].split('.')[0])
        existing_files.add((layout_id, power_id))

print(f"Found {len(existing_files)} existing processed files")

# Process all synthetic layouts
for i in range(400):
    # Get chip size from layout_params
    chip_size = 12  # Default size
    try:
        with open('./dataset_synthetic/layout_params.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if int(parts[0]) == i:
                    chip_size = int(parts[2])
                    break
    except FileNotFoundError:
        print(f"Warning: layout_params.csv not found, using default chip size of 12mm")
    
    # Calculate grid resolution based on chip size with bounds
    grid_resolution = min(max(int(64 * (chip_size / 12)), 32), 512)
    
    synthetic_file = f"Synthetic_Chiplet_{i}.flp"
    if not os.path.exists(synthetic_file):
        print(f"Warning: {synthetic_file} not found, skipping layout {i}")
        continue
        
    # Process each power variation
    for j in range(20):
        # Skip if this combination is already processed
        if (i, j) in existing_files:
            print(f"Skipping existing file for Layout {i}, Power {j}")
            continue
            
        power_file_name = f"Synthetic_Chiplet_{i}_Power{j}.ptrace"
        if not os.path.exists(power_file_name):
            print(f"Warning: {power_file_name} not found, skipping power variation {j}")
            continue

        # Rename the floorplan file to Chiplet_Core.flp for HotSpot
        os.rename(synthetic_file, "Chiplet_Core.flp")
        
        # Try different grid resolutions if the first one fails
        grid_resolutions = [grid_resolution, 64, 128, 256, 32]
        success = False
        
        for res in grid_resolutions:
            # Run HotSpot simulation with current grid resolution
            cmd = f"../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p {power_file_name} -steady_file Chiplet.steady -model_type grid -grid_rows {res} -grid_cols {res} -grid_steady_file Chiplet.grid.steady"
            
            tmr_start = time.time()
            result = os.system(cmd)
            tmr_end = time.time()
            
            print(f"Layout {i}, Power {j}, Grid {res}: {tmr_end - tmr_start}s")
            
            # Check if command succeeded and output files exist
            if result == 0 and os.path.exists("./data/Edge.csv"):
                success = True
                break
        
        if not success:
            print(f"Failed to simulate layout {i}, power {j} with all grid resolutions")
            # Restore original filename
            os.rename("Chiplet_Core.flp", synthetic_file)
            continue
            
        # Check if output files exist before renaming
        if os.path.exists("./data/Edge.csv"):
            edge_file = f"./data/Edge_{i}_{j}.csv"
            os.rename("./data/Edge.csv", edge_file)
        else:
            print(f"Warning: Edge.csv not generated for layout {i}, power {j}")
            
        if os.path.exists("./data/Temperature.csv"):
            temp_file = f"./data/Temperature_{i}_{j}.csv"
            os.rename("./data/Temperature.csv", temp_file)
        else:
            print(f"Warning: Temperature.csv not generated for layout {i}, power {j}")
            
        if os.path.exists("./data/Power.csv"):
            power_file = f"./data/Power_{i}_{j}.csv"
            os.rename("./data/Power.csv", power_file)
        else:
            print(f"Warning: Power.csv not generated for layout {i}, power {j}")
            
        # Restore original filename
        os.rename("Chiplet_Core.flp", synthetic_file)
        
    print(f"Completed layout {i}")
    
print("Thermal simulation complete for all synthetic layouts!")
