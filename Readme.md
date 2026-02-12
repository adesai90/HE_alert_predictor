#  (Get a name):
See Flowchat for plan of Action....

Directories: 
- Inputs: Input file used
- Fermi_Sample: Should have fermi sources with their lc's
- (optional) Fermi LAT ssdc folder currently in time series forecasting...
- Model_predictions - Contains all the outpus (text files for just predictions for multiple cases), seperate test directory for testing and debugging, Inputs to these should be LCs with Flux and Index values only (so that in the future this can be replaced by other data, possibly with one or 2 parameters. 
- codes:
    - Other : from_cluaide_fermi_data_main_flux_index_prediction-Copy1 (not used)
            fermi_data_run_alerts_prediction to predict number of alerts per year, but might be easier to just run predicition for individual fluxes
    - 
    - extrapolate_to_amego_check_work.ipynb to check if AMEGO can detect it!
- ???AMEGO_X_comparisions (Should contain Amego_x Sensitivity and relevant tools)
- 


Procedure:
1. Run fermi_data_initial_sample_selection to first make the sample selection which is then used for the study. 

2. Run fermi_lat_get_alert_data_from_lc to download the LCR data and create initial files,  

3. Run the .ipynb to test one seed. Note that this will make plots and save everything. Run the python code to test multiple seeds which will only make text files with predictions.

3. Run  extrapolate_to_amego_check_work.ipynb to read the individual csv files and check how many are actually giving results.

For questions contact [adesai.physics@gmail.com]
