import random
import os

# Parameters for synthetic dataset generation
Chip_Sizes = [12, 14, 16, 18]  
Chiplet_Counts = [4, 5, 6, 7, 8]  
Chiplet_Sizes = [1, 2, 3, 4, 5] 
Center = [3, 4, 5, 6, 7, 8, 9]  
Power = [1, 3, 5, 7, 9]  
Num = 20 

# Dir check and creation 
dataset_dir = "dataset_syn"
if not os.path.exists(f'./{dataset_dir}'):
    os.makedirs(f'./{dataset_dir}')

# TIM creation
def regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, region, info):
    """Create a region with a chiplet and surrounding TIM (thermal interface material)"""
    with open('Synthetic_Chiplet_' + str(i) + '.flp', 'a') as CoreRec:
        # Left of chiplet
        Width = CO_x0 - BL_x0
        Height = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(BL_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # Right of chiplet
        Width = BL_x0 + BL_w - CO_x0 - CO_w
        Height = BL_h
        if Width != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str((CO_x0 + CO_w) * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # Below the chiplet
        Width = CO_w
        Height = CO_y0 - BL_y0
        if Height != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(CO_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        # Above the chiplet
        Width = CO_w
        Height = BL_y0 + BL_h - CO_y0 - CO_h
        if Height != 0:
            count = count + 1
            CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Height * 1e-3) + " " + 
                          str(CO_x0 * 1e-3) + " " + str((CO_y0 + CO_h) * 1e-3) + " " + str(4e6) + " " + str(0.25) + "\n")
            info.append(1)
        
        CoreRec.write("Core" + str(region) + " " + str(CO_w * 1e-3) + " " + str(CO_h * 1e-3) + " " + 
                      str(CO_x0 * 1e-3) + " " + str(CO_y0 * 1e-3) + "\n")
        info.append(2)
    
    return count, info

layout_params = []

# Generation of 400 synthetic layouts
def layout_generation(num_layout: int = 50):
    for i in range(50, 100):
        count = 0
        info = []
        
        # Random parameter selection for current layout
        chiplet_count = random.choice(Chiplet_Counts)
        chip_size = random.choice(Chip_Sizes)
        chiplet_size = random.choice(Chiplet_Sizes)
        
        # New floorplan file
        with open('Synthetic_Chiplet_' + str(i) + '.flp', 'w') as f:
            pass  
        
        # For 4 chiplets: Quadrant based approach
        if chiplet_count == 4:
            CenterX = random.choice(Center)
            CenterY = random.choice(Center)
            
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
            
            if CenterV and CenterH:  
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
            # For more than 4 chiplets: Grid-based approach
            # Applying scaling factor for larger chip sizes
            scale_factor = chip_size / 12  
            
            grid_size = int(chiplet_count**0.5)
            if grid_size**2 < chiplet_count:
                grid_size += 1
            
            # Scaling
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
            
            # Placing chiplets in regions
            for region_idx, (BL_x0, BL_y0, BL_w, BL_h) in enumerate(regions):
                CenterV = []
                CenterH = []
                

                x_start = int(BL_x0)
                x_end = int(BL_x0 + BL_w - chiplet_size + 0.999)  
                y_start = int(BL_y0)
                y_end = int(BL_y0 + BL_h - chiplet_size + 0.999) 
                
                for j in range(x_start, x_end + 1):
                    CenterV.append(j)
                for j in range(y_start, y_end + 1):  
                    CenterH.append(j)

                
                if CenterV and CenterH:
                    CO_x0 = random.choice(CenterV)
                    CO_y0 = random.choice(CenterH)
                    CO_w = chiplet_size  
                    CO_h = chiplet_size 

                else:
                    # Fallback: place chiplet at region center or adjust chiplet size
                    if chiplet_size <= BL_w and chiplet_size <= BL_h:
                        CO_x0 = int(BL_x0 + (BL_w - chiplet_size) / 2)
                        CO_y0 = int(BL_y0 + (BL_h - chiplet_size) / 2)
                        CO_w = chiplet_size  
                        CO_h = chiplet_size 

                    else:
                        adjusted_size = min(BL_w, BL_h, chiplet_size)
                        CO_x0 = int(BL_x0)
                        CO_y0 = int(BL_y0)
                        CO_w = adjusted_size
                        CO_h = adjusted_size
                        print(f"Warning: Adjusted chiplet size to {adjusted_size} for layout {i}, region {region_idx+1}")

                count, info = regionCreate(BL_x0, BL_y0, BL_w, BL_h, CO_x0, CO_y0, CO_w, CO_h, i, count, region_idx + 1, info)

        
        # Generation of power trace files/layout
        for j in range(Num):
            with open('Synthetic_Chiplet_' + str(i) + '_Power' + str(j) + '.ptrace', 'w') as CorePower:
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
                
                for k in range(len(info)):
                    if info[k] == 1:
                        CorePower.write(str(0) + " ") 
                    elif info[k] == 2:
                        CorePower.write(str(random.choice(Power)) + " ")  
                CorePower.write("\n")
        
        # Storage
        layout_params.append({
            'id': i,
            'chiplet_count': chiplet_count,
            'chip_size': chip_size,
            'chiplet_size': chiplet_size
        })
        
        print(f"Generated layout {i}: {chiplet_count} chiplets, {chip_size}×{chip_size} mm chip, {chiplet_size}×{chiplet_size} mm chiplets")

    # Saving for reference
    with open(f'./{dataset_dir}/layout_params.csv', 'w') as f:
        f.write("id,chiplet_count,chip_size,chiplet_size\n")
        for params in layout_params:
            f.write(f"{params['id']},{params['chiplet_count']},{params['chip_size']},{params['chiplet_size']}\n")

    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    layout_generation(num_layout=100)
