from prediction_codes import *
import os,sys
import glob
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from concurrent.futures import ProcessPoolExecutor

import glob

flux_factor_for_alert= 50 # Fixed for now
leave_number_indices=30

# Note fixing path for my directory, please change this based on your installation
wdir = "/Users/aadesai1/Desktop/In_use/ML_work/Fermi_amego_alert_project/Main/Model_predictions/cache_folder/"
os.chdir(wdir)

num_alerts_files = glob.glob(f'../../Fermi_sample/alerts_per_year_flux_factor_{flux_factor_for_alert}/num_alerts_per_year_file_*')
alerts_per_bin_files = glob.glob(f'../../Fermi_sample/alerts_per_year_flux_factor_{flux_factor_for_alert}/alerts_flux_data_per_year_file_*')

print("Current working directory: ",os.getcwd())

src_names=[]
for path_name in num_alerts_files:
    src_names.append(path_name[-21:][:-4])


id_val = "v3"

def run_seed(seed):

    for src_index in range(len(src_names)): 
        os.chdir(wdir) 
        if src_index!=2: #Test one source
            continue
        src_dir = f'{wdir}/src_{src_names[src_index]}flux_factor_{flux_factor_for_alert}/'
        if os.path.isdir(src_dir)==False:
            os.mkdir(src_dir)

        set_seed(seed)
        src_dir = f'{wdir}/src_{src_names[src_index]}flux_factor_{flux_factor_for_alert}/seed_{seed}/'
        if os.path.isdir(src_dir)==False:
            os.mkdir(src_dir)
        
        alerts_data = np.load(alerts_per_bin_files[src_index])
        all_times = alerts_data[0]  # MJD times
        all_flux = np.log10(alerts_data[1])   # Flux values
        all_index = alerts_data[2]  # Index values
            
        # Create DataFrame with flux and index
        df_orig = pd.DataFrame({
                '#MJD': all_times,
                'flux': all_flux,
                'index': all_index
            })
        
        

        # Resample
        num_points = 300
        mjd_resampled = np.linspace(all_times.min(), all_times.max(), num_points)

        flux_interp = interp1d(all_times, all_flux, kind='linear')
        index_interp = interp1d(all_times, all_index, kind='linear')

        flux_resampled = flux_interp(mjd_resampled)
        index_resampled = index_interp(mjd_resampled)  

        #SPLIT INTO TRAIN AND TEST!
        test_split_idx = num_points - leave_number_indices  
        test_mjd = mjd_resampled[:test_split_idx]  
        test_flux = flux_resampled[:test_split_idx]  
        test_index = index_resampled[:test_split_idx]  
        
        df = pd.DataFrame({
            '#MJD': mjd_resampled,
            'flux': flux_resampled,
            'index': index_resampled
        })

        flux = flux_interp(mjd_resampled).reshape(-1,1)
        index = index_interp(mjd_resampled).reshape(-1,1)

        data = np.concatenate([flux, index], axis=1)

        flux_scaler = StandardScaler()
        index_scaler = StandardScaler()

        flux_scaled = flux_scaler.fit_transform(flux)
        index_scaled = index_scaler.fit_transform(index)

        data_scaled = np.concatenate([flux_scaled, index_scaled], axis=1)


        print(f"Data shape: {data_scaled.shape}")
        data_scaled_train = data_scaled[:test_split_idx] 

        seq_len = 12 
        forecast_horizon = leave_number_indices

        os.chdir(src_dir) 
        print(f'Running on {src_dir}')
        
        dataset = AstroLightcurveDataset(
            data_scaled_train,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon
        )
        print("Dataset creation complete!")

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )
        print(f"Dataset size: {len(dataset)}, Train: {train_size}, Val: {val_size}")

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32)

        #device = "cuda" if torch.cuda.is_available() else "cpu" 
        device = "cpu"

        model = AstroForecastModel(
            input_dim=2,            # flux + index
            hidden_dim=32,
            forecast_horizon=forecast_horizon,
            num_heads=2
        )


        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=120,
            lr=3e-4,
            device=device,
            patience=15  # Early stopping
        )

        print("Training done!")

        # ===== VALIDATION: Predict last forecast_horizon points =====
        # Get the sequence BEFORE the last forecast_horizon points
        last_sequence = data_scaled[test_split_idx - seq_len:test_split_idx]
        
    
        print(f"\nValidation prediction:")
        print(f"  Using sequence indices: {test_split_idx - seq_len} to {test_split_idx}")
        print(f"  Predicting indices: {test_split_idx} to {test_split_idx + forecast_horizon}")
    
        last_sequence = data_scaled[-seq_len:]

        forecast_results = forecast(
            model,
            last_sequence,
            flux_scaler,index_scaler,
            flux_is_log=True,
            device=device
        )

        forecast_results.to_csv(f'{src_dir}/forecast_results_{src_names[src_index]}_seed_{seed}.csv')
    print("All sources Ran!")
    return



if __name__ == "__main__":
    seeds = range(1, 100)   # seeds = [1, 2]

    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(run_seed, seeds)