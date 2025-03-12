import random
import os
import torch
import math
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch import Tensor
from scipy.ndimage import label, find_objects

Center = [3, 4, 5, 6, 7, 8, 9]
ChipW = 12
CoreW = 3
Power = [1, 3, 5, 7, 9]
Num = 20

def region_generate(
        layout_path: str, 
        num_grid_x: int = 100, # control the grain
        num_grid_y: int = 100, # control the grain
        numcore: int = 1,
        i: int = 0,
        mode: str = "dreamplace",
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ) -> None:

    assert mode in ['random', 'dreamplace'], f"{mode} not supported yet"

    if not os.path.exists(layout_path):
        os.mkdir(layout_path)

    num_core_grid_x = (CoreW / ChipW * num_grid_x).__floor__()
    num_core_grid_y = (CoreW / ChipW * num_grid_y).__floor__()

    # initialize a probability map
    proba_map = torch.ones(size=([num_grid_x - num_core_grid_x, num_grid_y - num_core_grid_y]))

    core_UL_row, core_UL_col = [], []

    for _ in range(numcore):
        non_zero_indices = proba_map.flatten() > 0
        filtered_weights = proba_map.flatten()[non_zero_indices]
        indices = torch.nonzero(non_zero_indices, as_tuple=True)[0]

        samples = WeightedRandomSampler(weights=filtered_weights, num_samples=1, replacement=True)
        for i in samples:
            idx = int(indices[i])
            # print(idx)
            rows, cols = (idx / proba_map.shape[1]).__floor__(), idx % proba_map.shape[1]

            core_UL_row.append(rows)
            core_UL_col.append(cols)

            center = [rows + (num_core_grid_x / 2).__floor__(), cols + (num_core_grid_x / 2).__floor__()]

            # Upate probability map according to placed cores
            proba_map = update_probability_map(
                center=center,
                proba_map=proba_map,
                alpha=10,
                beta=400
            )
            # proba_map = torch.nn.functional.softmax(proba_map.flatten() / Temp, dim=-1).reshape(map_shape)
            proba_map = (proba_map - proba_map.min()) * 10000
            if mode == 'dreamplace':
                for i in range(len(core_UL_row)):
                    proba_map[core_UL_row[i] - num_core_grid_x: core_UL_row[i] + num_core_grid_x, core_UL_col[i] - num_core_grid_y: core_UL_col[i] + num_core_grid_y] = 0

    return core_UL_row, core_UL_col, num_core_grid_x, num_core_grid_y


def update_probability_map(center: list, proba_map: Tensor, alpha: float, beta: float) -> Tensor:
    for i in range(proba_map.shape[0]):
        for j in range(proba_map.shape[1]):
            L1norm = math.sqrt(((i - center[1]) ** 2 + (j - center[0]) ** 2))
            proba_map[i, j] = -alpha * L1norm ** 2 + beta * L1norm

    return proba_map


def calculate_total_distance(core_UL_row: list, core_UL_col: list) -> float:
    n = len(core_UL_row)
    sum_x = sum(core_UL_row)
    sum_x2 = sum(x ** 2 for x in core_UL_row)

    sum_y = sum(core_UL_col)
    sum_y2 = sum(y ** 2 for y in core_UL_col)

    distance = n * sum_x2 - sum_x ** 2 + n * sum_y2 - sum_y ** 2
    return distance


def check_intersection(core_UL_row: list, core_UL_col: list, num_core_grid_x: int, num_core_grid_y: int, count: int = 0):
    for i in range(0, len(core_UL_row)):
        for j in range(i + 1, len(core_UL_row)):
            if (core_UL_row[i] - core_UL_row[j]).__abs__() < num_core_grid_x:
                if (core_UL_col[i] - core_UL_col[j]).__abs__() < num_core_grid_y:
                    print(f"Not pass check")
                    return count, None

    print(f"pass")
    count -= 1
    dis = calculate_total_distance([i * ChipW / 100 for i in core_UL_row], core_UL_col=[i * ChipW / 100 for i in core_UL_col])
    return count, dis


def dataset_generation(count: int = 1, mode: str = "random"):
    total_dis = 0
    total_count = count
    core_UL_row_list, core_UL_col_list = [], []
    while count > 0:
        core_UL_row, core_UL_col, num_core_grid_x, num_core_grid_y = region_generate(
            layout_path="./newlayout",
            num_grid_x=100,
            num_grid_y=100,
            numcore=4,
            mode=mode
        )
        count, dis = check_intersection(
            core_UL_row=core_UL_row,
            core_UL_col=core_UL_col,
            num_core_grid_x=num_core_grid_x,
            num_core_grid_y=num_core_grid_y,
            count=count
        )
        print(count)

        if dis is not None:
            total_dis += dis
            core_UL_row_list.append(core_UL_row)
            core_UL_col_list.append(core_UL_col)

    regionCreate(
        core_UL_row_list=core_UL_row_list,
        core_UL_col_list=core_UL_col_list,
        num_grid_x=100,
        num_grid_y=100,
        num_core_grid_x=num_core_grid_x,
        num_core_grid_y=num_core_grid_y,
        i=count
    )
    
    print(total_dis / total_count)
    return core_UL_row_list, core_UL_col_list, total_dis / total_count


def regionCreate(
        core_UL_row_list: list[list], 
        core_UL_col_list: list[list], 
        num_grid_x: int, 
        num_grid_y: int, 
        num_core_grid_x: int,
        num_core_grid_y: int,
        i: int
    ):
    with open('newlayout/Chiplet_Core' + str(i) + '.flp', 'a') as CoreRec:
        for ite in range(len(core_UL_row_list)):
            count = 0
            
            chip_fp = torch.zeros(size=[num_grid_x, num_grid_y])

            TL_index_X, TL_index_Y = core_UL_row_list[ite], core_UL_col_list[ite]     # is a list
            print(TL_index_X, TL_index_Y)

            for p in range(len(TL_index_X)):
                chip_fp[TL_index_X[p]: TL_index_X[p] + num_core_grid_x, TL_index_Y[p]: TL_index_Y[p] + num_core_grid_y] = 1

            
        TIM_index = chip_fp.nonzero()
        print(TIM_index)

        # if Width != 0:
        #     count = count + 1
        #     CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(BL_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
        #     info.append(1)
        # Width = BL_x0 + BL_w - CO_x0 - CO_w
        # Hight = BL_h
        # if Width != 0:
        #     count = count + 1
        #     CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str((CO_x0 + CO_w) * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
        #     info.append(1)
        # Width = CO_w
        # Hight = CO_y0 - BL_y0
        # if Hight != 0:
        #     count = count + 1
        #     CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str(BL_y0 * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
        #     info.append(1)
        # Width = CO_w
        # Hight = BL_y0 + BL_h - CO_y0 - CO_h
        # if Hight != 0:
        #     count = count + 1
        #     CoreRec.write("TIM" + str(count) + " " + str(Width * 1e-3) + " " + str(Hight * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str((CO_y0 + CO_h) * 1e-3) + " " + str(4e6)+" "+str(0.25)+"\n")
        #     info.append(1)
        # CoreRec.write("Core" + str(region) + " " + str(CO_w * 1e-3) + " " + str(CO_h * 1e-3) + " " + str(CO_x0 * 1e-3) + " " + str(CO_y0 * 1e-3)+"\n")
        # info.append(2)

        return None

if __name__ == "__main__":
    torch.random.manual_seed(1234)
    # region_generate(
    #     layout_path="./newlayout",
    #     num_grid_x=100,
    #     num_grid_y=100,
    #     numcore=4
    # )
    dataset_generation(count = 1, mode="random")
    # nonzero_blocks, top_left_corners = slicing_chip(
    #     core_UL_row=[1, 5, 4, 8],
    #     core_UL_col=[3, 4, 1, 7],
    #     num_grid_x=10,
    #     num_grid_y=10,
    #     num_core_grid_y=2,
    #     num_core_grid_x=2,
    #     count=0
    # )

    # for block, corner in zip(nonzero_blocks, top_left_corners):
    #     print("Nonzero block:")
    #     print(block)
    #     print("Top-left corner:", corner)