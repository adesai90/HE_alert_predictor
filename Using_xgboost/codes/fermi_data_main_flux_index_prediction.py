from prediction_codes import *
from concurrent.futures import ProcessPoolExecutor

import glob

flux_factor_for_alert= 50 # Fixed for now
leave_number_indices=-6

# Note fixing path for my directory, please change this based on your installation
wdir = "/Users/aadesai1/Desktop/In_use/ML_work/Fermi_amego_alert_project/Main/Using_xgboostModel_predictions/cache_folder/"
os.chdir(wdir)

num_alerts_files = glob.glob(f'../../Fermi_sample/alerts_per_year_flux_factor_{flux_factor_for_alert}/num_alerts_per_year_file_*')
alerts_per_bin_files = glob.glob(f'../../Fermi_sample/alerts_per_year_flux_factor_{flux_factor_for_alert}/alerts_flux_data_per_year_file_*')

print("Current working directory: ",os.getcwd())

src_names=[]
for path_name in num_alerts_files:
    src_names.append(path_name[-21:][:-4])


id_val = "v3"

def run_seed(seed):
        if os.path.isdir(f'seed_{seed}_flux_factor_{flux_factor_for_alert}/')!=True:
            os.mkdir(f'seed_{seed}_flux_factor_{flux_factor_for_alert}/')

        for src_index in range(len(src_names)): 
            print(f"\n{'='*60}")
            print(f"Training models for {src_names[src_index]}")
            print(f"{'='*60}")

            train_model(wdir,
                        src_names[src_index], 
                        seed,
                        flux_factor_for_alert,
                        leave_number_indices, 
                        id_val, 
                        model_type='flux')

            train_model(wdir,
                        src_names[src_index], 
                        seed,
                        flux_factor_for_alert, 
                        leave_number_indices,
                        id_val, 
                        model_type='index')

            # This will create model files per source for both flux and index

            save_directory = f'/Users/aadesai1/Desktop/In_use/ML_work/Fermi_amego_alert_project/Main/Model_predictions/predictions/seed_{seed}'
            if not os.path.isdir(save_directory):
                os.mkdir(save_directory)

            # Next we want the prediction!
            make_predictions_data(wdir,
                          seed,
                          flux_factor_for_alert,
                          src_names[src_index], 
                          save_directory, 
                          id_val, 
                          model_type='flux')

            make_predictions_data(wdir,
                          seed,
                          flux_factor_for_alert,
                          src_names[src_index], 
                          save_directory, 
                          id_val, 
                          model_type='index')

if __name__ == "__main__":

    # First create Feature files that will be used for all sources 
    create_feature_files(wdir,
                        src_names, 
                        alerts_per_bin_files, 
                        flux_factor_for_alert,
                        id_val)


    

    seeds = range(1, 3)   # seeds = [1, 2]

    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.map(run_seed, seeds)