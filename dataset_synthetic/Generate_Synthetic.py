import random
import os

# Parameters for synthetic dataset generation
Chip_Sizes = [12, 14, 16, 18]  # Chip dimensions in mm
Chiplet_Counts = [4, 5, 6, 7, 8]  # Number of chiplets per layout
Chiplet_Sizes = [1, 2, 3, 4, 5]  # Chiplet dimensions in mm
Center = [3, 4, 5, 6, 7, 8, 9]  # Center points for region division
Power = [1, 3, 5, 7, 9]  # Power values in W
Num = 20  # Power variations per layout

# Create directory for synthetic dataset if it doesn't exist
if not os.path.exists('./dataset_synthetic'):
    os.makedirs('./dataset_synthetic')

def regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, region, info):
    """Create a region with a chiplet and surrounding TIM (thermal interface material)"""
    with open('Synthetic_Chiplet_' + str(i) + '.flp', 'a') as CoreRec:
        # TIM to the left of chiplet
        Width = CO_x0 - BL_x0
        Height = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(BL_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # TIM to the right of chiplet
        Width = BL_x0 + BL_w - CO_x0 - CO_w
        Height = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str((CO_x0 + CO_w) * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # TIM below chiplet
        Width = CO_w
        Height = CO_y0 - BL_y0
        if Height != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(CO_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # TIM above chiplet
        Width = CO_w
        Height = BL_y0 + BL_h - CO_y0 - CO_h
        if Height != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(CO_x0 * 1e-3) + " " + str((CO_y0 + CO_h) * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # Write chiplet
        CoreRec.write("Core" + str(region) + " " + str(CO_w * 1e-3) + " " + str(CO_h * 1e-3) + " " + 
                      str(CO_x0 * 1e-3) + " " + str(CO_y0 * 1e-3) + "\n")
        info.append(2)
    
    return count, info

# Store layout parameters for reference
layout_params = []

# Generate 400 synthetic layouts
for i in range(400):
    count = 0
    info = []
    
    # Randomly select parameters for this layout
    chiplet_count = random.choice(Chiplet_Counts)
    chip_size = random.choice(Chip_Sizes)
    chiplet_size = random.choice(Chiplet_Sizes)
    
    # Create new floorplan file
    with open('Synthetic_Chiplet_' + str(i) + '.flp', 'w') as f:
        pass  # Just create an empty file to start
    
    # For 4 chiplets, use the original region division approach
    if chiplet_count == 4:
        CenterX = random.choice(Center)
        CenterY = random.choice(Center)
        
        # Adjust center points based on chip size
        if chip_size != 12:
            scale_factor = chip_size / 12
            CenterX = int(CenterX * scale_factor)
            CenterY = int(CenterY * scale_factor)
        
        # Region 1 (bottom-left)
        BL_x0 = 0
        BL_y0 = 0
        BL_w = CenterX
        BL_h = CenterY
        
        CenterV = []
        CenterH = []
        for j in range(BL_x0, BL_x0 + BL_w - chiplet_size + 1):
            CenterV.append(j)
        for j in range(BL_y0, BL_y0 + BL_h - chiplet_size + 1):
            CenterH.append(j)
        
        if CenterV and CenterH:  # Ensure lists are not empty
            CO_x0 = random.choice(CenterV)
            CO_y0 = random.choice(CenterH)
            CO_w = chiplet_size
            CO_h = chiplet_size
            count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 1, info)
        
        # Region 2 (bottom-right)
        BL_x0 = CenterX
        BL_y0 = 0
        BL_w = chip_size - CenterX
        BL_h = CenterY
        
        CenterV = []
        CenterH = []
        for j in range(BL_x0, BL_x0 + BL_w - chiplet_size + 1):
            CenterV.append(j)
        for j in range(BL_y0, BL_y0 + BL_h - chiplet_size + 1):
            CenterH.append(j)
        
        if CenterV and CenterH:
            CO_x0 = random.choice(CenterV)
            CO_y0 = random.choice(CenterH)
            CO_w = chiplet_size
            CO_h = chiplet_size
            count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 2, info)
        
        # Region 3 (top-right)
        BL_x0 = CenterX
        BL_y0 = CenterY
        BL_w = chip_size - CenterX
        BL_h = chip_size - CenterY
        
        CenterV = []
        CenterH = []
        for j in range(BL_x0, BL_x0 + BL_w - chiplet_size + 1):
            CenterV.append(j)
        for j in range(BL_y0, BL_y0 + BL_h - chiplet_size + 1):
            CenterH.append(j)
        
        if CenterV and CenterH:
            CO_x0 = random.choice(CenterV)
            CO_y0 = random.choice(CenterH)
            CO_w = chiplet_size
            CO_h = chiplet_size
            count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 3, info)
        
        # Region 4 (top-left)
        BL_x0 = 0
        BL_y0 = CenterY
        BL_w = CenterX
        BL_h = chip_size - CenterY
        
        CenterV = []
        CenterH = []
        for j in range(BL_x0, BL_x0 + BL_w - chiplet_size + 1):
            CenterV.append(j)
        for j in range(BL_y0, BL_y0 + BL_h - chiplet_size + 1):
            CenterH.append(j)
        
        if CenterV and CenterH:
            CO_x0 = random.choice(CenterV)
            CO_y0 = random.choice(CenterH)
            CO_w = chiplet_size
            CO_h = chiplet_size
            count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, 4, info)
    
    else:
        # For more than 4 chiplets, use a grid-based approach
        # Apply scaling factor for larger chip sizes
        scale_factor = chip_size / 12  # Base scale factor on 12mm reference
        
        grid_size = int(chiplet_count**0.5)
        if grid_size**2 < chiplet_count:
            grid_size += 1
        
        # Scale cell_size based on chip_size
        cell_size = chip_size / grid_size
        
        # Create regions based on grid
        regions = []
        for row in range(grid_size):
            for col in range(grid_size):
                if len(regions) < chiplet_count:
                    regions.append((
                        col * cell_size,  # x0
                        row * cell_size,  # y0
                        cell_size,        # width
                        cell_size         # height
                    ))
        
        # Place chiplets in regions
        for region_idx, (BL_x0, BL_y0, BL_w, BL_h) in enumerate(regions):
            CenterV = []
            CenterH = []
            
            # Explicitly convert floating-point boundaries to integers with floor/ceiling as appropriate
            x_start = int(BL_x0)
            x_end = int(BL_x0 + BL_w - chiplet_size + 0.999)  # Adding 0.999 to round up
            y_start = int(BL_y0)
            y_end = int(BL_y0 + BL_h - chiplet_size + 0.999)  # Adding 0.999 to round up
            
            for j in range(x_start, x_end + 1):  # +1 to include x_end
                CenterV.append(j)
            for j in range(y_start, y_end + 1):  # +1 to include y_end
                CenterH.append(j)

            
            if CenterV and CenterH:
                CO_x0 = random.choice(CenterV)
                CO_y0 = random.choice(CenterH)
                CO_w = chiplet_size  # Add this line
                CO_h = chiplet_size 

            else:
                # Fallback: place chiplet at region center if possible, or adjust chiplet size
                if chiplet_size <= BL_w and chiplet_size <= BL_h:
                    # Place at center of region
                    CO_x0 = int(BL_x0 + (BL_w - chiplet_size) / 2)
                    CO_y0 = int(BL_y0 + (BL_h - chiplet_size) / 2)
                    CO_w = chiplet_size  # Add this line
                    CO_h = chiplet_size 

                else:
                    # Adjust chiplet size to fit region
                    adjusted_size = min(BL_w, BL_h, chiplet_size)
                    CO_x0 = int(BL_x0)
                    CO_y0 = int(BL_y0)
                    CO_w = adjusted_size
                    CO_h = adjusted_size
                    print(f"Warning: Adjusted chiplet size to {adjusted_size} for layout {i}, region {region_idx+1}")

            # Add after both the if/else blocks for chiplet placement
            count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, region_idx + 1, info)

    
    # Generate power trace files for each layout
    for j in range(Num):
        with open('Synthetic_Chiplet_' + str(i) + '_Power' + str(j) + '.ptrace', 'w') as CorePower:
            # Write header
            temp1 = 0
            temp2 = 0
            for k in range(len(info)):
                if info[k] == 1:
                    temp1 = temp1 + 1
                    CorePower.write("TIM" + str(temp1) + " ")
                elif info[k] == 2:
                    temp2 = temp2 + 1
                    CorePower.write("Core" + str(temp2) + " ")
            CorePower.write("\n")
            
            # Write power values
            for k in range(len(info)):
                if info[k] == 1:
                    CorePower.write(str(0) + " ")  # TIM has 0 power
                elif info[k] == 2:
                    CorePower.write(str(random.choice(Power)) + " ")  # Random power for chiplets
            CorePower.write("\n")
    
    # Store layout parameters
    layout_params.append({
        'id': i,
        'chiplet_count': chiplet_count,
        'chip_size': chip_size,
        'chiplet_size': chiplet_size
    })
    
    print(f"Generated layout {i}: {chiplet_count} chiplets, {chip_size}×{chip_size} mm chip, {chiplet_size}×{chiplet_size} mm chiplets")

# Save layout parameters for reference
with open('./dataset_synthetic/layout_params.csv', 'w') as f:
    f.write("id,chiplet_count,chip_size,chiplet_size\n")
    for params in layout_params:
        f.write(f"{params['id']},{params['chiplet_count']},{params['chip_size']},{params['chiplet_size']}\n")

print("Synthetic dataset generation complete!")
