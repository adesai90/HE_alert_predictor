#  Predicting AMEGO alerts using Fermi LAT Gamma-ray data:
See Flowchat for plan of Action....

Directories: 
- Fermi_Sample: Should have fermi sources with their lc's
- (optional) Fermi LAT ssdc folder currently in time series forecasting...
- Model_predictions - Contains all the outpus (text files for just predictions for multiple cases), seperate test directory for testing and debugging, Inputs to these should be LCs with Flux and Index values only (so that in the future this can be replaced by other data, possibly with one or 2 parameters. 
- codes:
    - Other : from_cluaide_fermi_data_main_flux_index_prediction-Copy1 (not used)
            fermi_data_run_alerts_prediction to predict number of alerts per year, but might be easier to just run predicition for individual fluxes
    - fermi_lat_get_alert_data_from_lc.ipynb: Code to create alert lists based on particular flux threshold
    - fermi_data_main_flux_index_prediction: Main code to test prediction setup
    - extrapolate_to_amego_check_work.ipynb to check if AMEGO can detect it!
- ???AMEGO_X_comparisions (Should contain Amego_x Sensitivity and relevant tools)
- 


Procedure:
1. Run fermi_data_initial_sample_selection to first make the sample selection which is then used for the study. 

2. Run fermi_lat_get_alert_data_from_lc.ipynb to to make alert files that will then be used by the next code to test or run, This will create multiple numpy files for every source based on the condition of flux 
    Based on a flux threshold:
    - num_alerts_per_year_file give number of alerts per year [num_alerts_per_indi_bin]
    - alerts_flux_data_per_year_file_ gives additional information [times,fluxes,indices]

3. For Running: fermi_data_main_flux_index_prediction.py to sue multiple seeds to give predictions that will then be saved.
    -FOR TESTING ONLY!:Run the fermi_data_main_flux_index_prediction.ipynb to test one seed. Note that this will make plots and save everything. Run the python code to test multiple seeds which will only make text files with predictions.

4. Run  extrapolate_to_amego_check_work.ipynb to read the individual csv files and check how many are actually giving results.

For questions contact [adesai.physics@gmail.com]
