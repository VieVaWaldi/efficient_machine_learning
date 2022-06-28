import time
import datetime
import os

""" 
    Simple script that performs grid search for unet:
    1. grid_unet_slurm.job starts this script
    2. This script
        - Sets hyper para search space
        - Runs a slurm job for milan with unet
        - Waits for return and notes para and results -> appends to csv and closes!
    3. Repeat -> Profit

    Hyperparameter:
    * run_name: grid_unet_dd:mm_hh:mm
    * n_epochs: 40

    - n_init_channels   = 64 [32 - 84]
    - n_levels          = 2 [2 - 6]

    - lr                = 1E-4 (make adaptive)
    - l_optimizer       = torch.optim.Adam( l_unet2d.parameters(), lr )
    - l_loss_func       = torch.nn.CrossEntropyLoss()

    Results are stored in a csv.
    - final_test_accuracy
    - final_test_loss
"""

parameters = ["run_name", "n_epochs", "n_init_channels", "n_levels", 
    "lr", "l_optimizer", "l_loss_func"]
results = ["final_test_accuracy", "final_test_loss", "time"]

now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')
csv_name = f"csv/grid_search_result_{now}.csv"
csv_del = ", "
csv_eol = "\n"
csv_header = f"{csv_del.join(parameters)} {csv_del} {csv_del.join(results)} {csv_eol}"


with open(csv_name, 'a') as f:
    f.write(csv_header)    


run_name = f"grid_unet_{now}"
n_epochs = 50

lr = 1E-4
l_optimizer = "AdamW"
l_loss_func = "CrossEntropy"

def poll_run_finished():
    pass

for n_init_channels in range(60, 80, 5):
    for n_levels in range(3, 6, 1):
        run_cmd = f"sbatch unet_slurm.job {run_name} {n_epochs} {n_init_channels} {n_levels} {lr} {l_optimizer} {l_loss_func} {csv_name}"
        os.system(run_cmd)

        time_hours = 2*60*60
        time.sleep(time_hours)

        # poll 

