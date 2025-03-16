# T2GCN

T2GCN is the trasnferable ThermalGCN, a fast graph convolutional networks(GCN)-based method for thermal simulation of chiplet-based systems.

- Use global information (total power) as input,
- Apply the skip connection in graph convolution network,
- Integrate PNA network into the model,
- Use edge based attention network to represent the connection effect,
- Utilize more generalized dataset synthesis generation algorithms.

## Installation

ThermalGCN or T2GCN require Pytorch and DGL to be installed as backend using ```conda``` or ```pip```.
All other missing packages can also be install with ```conda``` or ```pip```.


## Instructions
- Chiplet layout generation:
  
  - Random (Original dataset)
    
    ```cd ./dataset_original/```
    
    ```python Generate.py```

  - Scalable (Synthetic dataset)
  
    ```cd ./dataset_synthetic/```
    
    ```python Generate_Synthetic.py```

  - Placement-aware
    
    ```cd ./dataset_special/```
    
    ```python layout_generation.py```



- Obtain dataset:

  In each sub dataset (dataset_original etc.)

  ```python run.py```
  
  to run hotspot and generate dataset which is stored into "./dataset_xxx/data".
  
  ```python data_preprocess.py```
  
  to normalize the data.

- Training GCN or GCNPNAGAT:
  
  ```python GCN.py```

  ```python GCNPNAGAT.py```

- Fine-tuning the model from pre-trained model:
  
  ```python3 GCN_tuning.py```

## Publications

L. Chen, W. Jin and S. X.-D. Tan, "Fast Thermal Analysis for Chiplet Design based on Graph Convolution Networks," 2022 27th Asia and South Pacific Design Automation Conference (ASP-DAC), 2022, pp. 485-492..

## The Team

ThermalGCN was originally developed by [Liang Chen](https://vsclab.ece.ucr.edu/people/liang-chen) and [Wentian Jin](https://vsclab.ece.ucr.edu/people/wentian-jin) at [VSCLAB](https://vsclab.ece.ucr.edu/VSCLAB) under the supervision of Prof. [Sheldon Tan](https://profiles.ucr.edu/app/home/profile/sheldont).

ThermalGCN is currently maintained by [Liang Chen](https://vsclab.ece.ucr.edu/people/liang-chen).

T2GCN is currently developed by [Haotian Lu](https://github.com/TRIGGERONE) and [Sai Surya Vidul Chinthamaneni](https://github.com/Mathio11).

