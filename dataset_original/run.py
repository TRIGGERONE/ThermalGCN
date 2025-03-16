import os
import time

data_dir = "./data"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

for i in range(50):
    ChipletCoreFile = "Chiplet_Core"+str(i)+".flp"
    os.rename(ChipletCoreFile, "Chiplet_Core.flp")
    for j in range(20):
        cmd = "../hotspot -c ./hotspot.config -f Chiplet_Core.flp -p Chiplet_Core"+str(i)+"_Power"+str(j)+".ptrace -steady_file Chiplet.steady  -model_type grid -detailed_3D on -grid_layer_file Chiplet.lcf -grid_steady_file Chiplet.grid.steady"
        tmr_start = time.time()
        os.system(cmd)
        tmr_end = time.time()
        print(tmr_end - tmr_start)

        print(i,j)

        
        EdgeFile = f"{data_dir}/Edge"+"_"+str(i)+"_"+str(j)+".csv"
        os.rename(f"{data_dir}/Edge.csv", EdgeFile)
        TempFile = f"{data_dir}/Temperature"+"_"+str(i)+"_"+str(j)+".csv"
        os.rename(f"{data_dir}/Temperature.csv", TempFile)
        PowerFile = f"{data_dir}/Power"+"_"+str(i)+"_"+str(j)+".csv"
        os.rename(f"{data_dir}/Power.csv", PowerFile)


    os.rename("Chiplet_Core.flp", ChipletCoreFile)
    
