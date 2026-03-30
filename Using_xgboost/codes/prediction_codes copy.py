import os,sys
import numpy as np
import pandas as pd
from astropy.io import fits

import joblib
import matplotlib.pyplot as plt

# For training:
import xgboost as xgb

import glob
from collections import deque

def create_feature_files(wdir,
                         src_names, 
                         alerts_per_bin_files, 
                         flux_factor_for_alert,
                         id_val):
    os.chdir(wdir)
    for src_index in range(len(src_names)):  
        directory = f'{wdir}/feature_store_flux_factor_{flux_factor_for_alert}'
        if os.path.isfile(f'{directory}/{src_names[src_index]}_index_features_{id_val}.csv')==True and \
            os.path.isfile(f'{directory}/{src_names[src_index]}_index_targets_{id_val}.csv')==True and \
            os.path.isfile(f'{directory}/{src_names[src_index]}_flux_features_{id_val}.csv')==True and \
            os.path.isfile(f'{directory}/{src_names[src_index]}_flux_targets_{id_val}.csv')==True:
            continue

        alerts_data = np.load(alerts_per_bin_files[src_index])
        all_times = alerts_data[0]  # MJD times
        all_flux = np.log10(alerts_data[1])   # Flux values
        all_index = alerts_data[2]  # Index values
        
        # Create DataFrame with flux and index
        df = pd.DataFrame({
            '#MJD': all_times,
            'flux': all_flux,
            'index': all_index
        })

        df.set_index('#MJD', inplace=True)
        
        print(f"\n{'='*70}")
        print(f"Processing: {src_names[src_index]}")
        print(f"{'='*70}")
        print(f"Total data points: {len(df)}")
        print(f"Non-zero flux points: {(df['flux'] > 0).sum()} ({100*(df['flux'] > 0).sum()/len(df):.1f}%)")
        print(f"Non-zero index points: {(df['index'] != 0).sum()} ({100*(df['index'] != 0).sum()/len(df):.1f}%)")

        # ---- CREATE FEATURES - USE PERCENTAGE CHANGE FOR FLUX ----
        batch_df = pd.DataFrame(index=df.index)
        
        # Add lagging Features 
        batch_df['flux_lag_1'] = df['flux'].shift(1) # Prediction 1 Use numdays: -1 day 0-1 = 1
        batch_df['flux_lag_2'] = df['flux'].shift(2) # Prediction 2 Use numdays: -2 day 0-2 = 2
        batch_df['flux_lag_3'] = df['flux'].shift(3) # Prediction 3 Use numdays: -3 day 0-4 = 3
        batch_df['flux_lag_4'] = df['flux'].shift(4) # 
        batch_df['flux_lag_5'] = df['flux'].shift(5) # 
        batch_df['flux_lag_6'] = df['flux'].shift(6) # 
    
        batch_df['index_lag_1'] = df['index'].shift(1) # Prediction 1 Use numdays: -1 day 0-1 = 1
        batch_df['index_lag_2'] = df['index'].shift(2) # Prediction 2 Use numdays: -2 day 0-2 = 2
        batch_df['index_lag_3'] = df['index'].shift(3) # Prediction 3 Use numdays: -3 day 0-4 = 3
        batch_df['index_lag_4'] = df['index'].shift(4) # 
        batch_df['index_lag_5'] = df['index'].shift(5) # 
        batch_df['index_lag_6'] = df['index'].shift(6) # 
        
        # Rolling statistics
        batch_df['flux_rolling_mean_7'] = df['flux'].rolling(window=7).mean().round(2)
        batch_df['flux_rolling_std_7'] = df['flux'].rolling(window=7).std().round(2)
        
        batch_df['index_rolling_mean_7'] = df['index'].rolling(window=7).mean().round(2)
        batch_df['index_rolling_std_7'] = df['index'].rolling(window=7).std().round(2)
        
        # Lagging target variable
        batch_df['flux_target_1d'] = df['flux'].shift(-1) # Next day
        batch_df['index_target_1d'] = df['index'].shift(-1) # Next day

        batch_df['flux_target_2d'] = df['flux'].shift(-2) # Next day
        batch_df['index_target_2d'] = df['index'].shift(-2) # Next day
        
        
        # Drop NaN rows
        batch_df = batch_df.dropna()
        
        if len(batch_df) < 5:
            print(f"Warning: {src_names[src_index]} has insufficient data")
            #continue
        
        # Create feature and target dataframes
        directory = f'{wdir}/feature_store_flux_factor_{flux_factor_for_alert}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Features for FLUX prediction
        flux_features_cols = [
            'flux_lag_1', 'flux_lag_2','flux_lag_3','flux_lag_4','flux_lag_5','flux_lag_6', 
            'flux_rolling_mean_7','flux_rolling_std_7', 
            'index_lag_1', 'index_lag_2','index_lag_3','index_lag_4','index_lag_5','index_lag_6', 
            'index_rolling_mean_7','index_rolling_std_7'
        ]
        flux_features_df = batch_df[flux_features_cols].copy()
        flux_features_df.to_csv(f'{directory}/{src_names[src_index]}_flux_features_{id_val}.csv')
        
        # Targets for FLUX prediction
        flux_targets_df = batch_df[['flux_target_1d','flux_target_2d']].copy()
        flux_targets_df.to_csv(f'{directory}/{src_names[src_index]}_flux_targets_{id_val}.csv')
        
        # Features for INDEX prediction
        index_features_cols = [
            'index_lag_1', 'index_lag_2','index_lag_3','index_lag_4','index_lag_5','index_lag_6', 
            'index_rolling_mean_7','index_rolling_std_7',
            'flux_lag_1', 'flux_lag_2','flux_lag_3','flux_lag_4','flux_lag_5','flux_lag_6', 
            'flux_rolling_mean_7','flux_rolling_std_7'
        ]
        index_features_df = batch_df[index_features_cols].copy()
        index_features_df.to_csv(f'{directory}/{src_names[src_index]}_index_features_{id_val}.csv')
        
        # Targets for INDEX prediction
        index_targets_df = batch_df[['index_target_1d', 'index_target_2d']].copy()
        index_targets_df.to_csv(f'{directory}/{src_names[src_index]}_index_targets_{id_val}.csv')
        
        print(f"Created feature files for {src_names[src_index]} - {len(batch_df)} samples")

def train_model(wdir,
                fermi_src,
                seed,
                flux_factor_for_alert,
                leave_number_indices, 
                id_val, 
                model_type='flux'):
    """
    Train XGBoost model - 
    """
    os.chdir(wdir)
    filename = f'{wdir}/seed_{seed}_flux_factor_{flux_factor_for_alert}/models/{fermi_src}_{model_type}_model_{id_val}.pkl'
    if os.path.isfile(filename):
        print(f"Model already exists: {filename}")
        return

    feature_directory = f'{wdir}/feature_store_flux_factor_{flux_factor_for_alert}'
    feature_file = f'{feature_directory}/{fermi_src}_{model_type}_features_{id_val}.csv'
    target_file = f'{feature_directory}/{fermi_src}_{model_type}_targets_{id_val}.csv'
    
    X_train = pd.read_csv(feature_file)
    X_train.set_index('#MJD', inplace=True)
    
    targets_df = pd.read_csv(target_file)
    targets_df.set_index('#MJD', inplace=True)
 
    # Use all data except last points for training
    X_train_mod = X_train.head(leave_number_indices)
    targets_df_mod = targets_df.head(leave_number_indices)
    
    # Train on target (percentage for flux, absolute for index)
    y_train = targets_df_mod[f'{model_type}_target_1d']

    # Remove any remaining NaN values
    #mask = ~(X_train_mod.isna().any(axis=1) | y_train.isna())
    #X_train_mod = X_train_mod[mask]
    #y_train = y_train[mask]
    
    if len(X_train_mod) < 5:
        print(f"Warning: {fermi_src} has insufficient data for {model_type} training")
        #return
    
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} model for {fermi_src}")
    print(f"  Target: {'target_1d'}")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train_mod)}")
    print(f"Target statistics:")
    print(f"  Min:  {y_train.min():.6f}")
    print(f"  Max:  {y_train.max():.6f}")
    print(f"  Mean: {y_train.mean():.6f}")
    print(f"  Std:  {y_train.std():.6f}")
    
    # Model parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.3,
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'tree_method': 'hist',
    }
    
    dtrain = xgb.DMatrix(data=X_train_mod, label=y_train)
    
    # Cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=200,
        nfold=3,
        early_stopping_rounds=30,
        metrics='rmse',
        as_pandas=True,
        seed=seed,
        verbose_eval=False
    )
    
    optimal_rounds = max(cv_results['test-rmse-mean'].idxmin() + 1, 20)
    best_rmse = cv_results['test-rmse-mean'].min()
    
    print(f"\nCross-validation results:")
    print(f"  Best RMSE: {best_rmse:.6f}")
    print(f"  Optimal boost rounds: {optimal_rounds}")
    
    # Train final model
    final_model = xgb.XGBRegressor(
        n_estimators=optimal_rounds,
        **params,
        random_state=seed
    )
    
    final_model.fit(X_train_mod, y_train)
    
    # Check predictions
    train_pred = final_model.predict(X_train_mod)
    pred_std = np.std(train_pred)
    pred_range = train_pred.max() - train_pred.min()
    pred_mean = np.mean(train_pred)
    
    print(f"\nTraining predictions:")
    print(f"  Min:   {train_pred.min():.6f}")
    print(f"  Max:   {train_pred.max():.6f}")
    print(f"  Mean:  {pred_mean:.6f}")
    print(f"  Std:   {pred_std:.6f}")
    print(f"  Range: {pred_range:.6f}")
    
    
    # Save model
    if not os.path.isdir(f'{wdir}/seed_{seed}_flux_factor_{flux_factor_for_alert}/models'):
        os.mkdir(f'{wdir}/seed_{seed}_flux_factor_{flux_factor_for_alert}/models')
    
    joblib.dump(final_model, filename)
    
    print(f"\n✓ Saved model: {filename}")
    print(f"{'='*70}\n")
    
    del dtrain, cv_results


def make_predictions_data(wdir,
                          seed,
                          flux_factor_for_alert,
                          src, 
                          save_directory, 
                          id_val, 
                          model_type='flux'):
    """
    Make predictions and reconstruct absolute values
    """
    filename = f'{save_directory}/seed_{seed}_fl_thresh_{flux_factor_for_alert}_{src}_{model_type}_recursive_predictions.npy'
    if os.path.isfile(filename):
        print(f"Model already exists: {filename}")
        return
    
    try:
        print(f"\n{'='*70}")
        print(f"Making {model_type.upper()} predictions for {src}")
        print(f"{'='*70}")
            
        # Load features
        feature_directory = f'{wdir}/feature_store_flux_factor_{flux_factor_for_alert}'
    
        features_df = pd.read_csv(f'{feature_directory}/{src}_{model_type}_features_{id_val}.csv')
        features_df.set_index('#MJD', inplace=True)
        #features_df = features_df.head(leave_number_indices)
        features_df = features_df.dropna()
            
        # Load model and predict 
        model_file = f'{wdir}/seed_{seed}_flux_factor_{flux_factor_for_alert}/models/{src}_{model_type}_model_{id_val}.pkl'
        model = joblib.load(model_file)
        
        
        # --------------------------------------------------
        # Recursive forecasting setup (derived using suggestions)
        # --------------------------------------------------
        N_future = 50   # number of future steps you want

        current_features = features_df.copy()
        future_predictions = []
        future_times = []

        last_time = current_features.index[-1]

        # identify lag columns automatically
        lag_cols = [c for c in current_features.columns if f'{model_type}_lag' in c]

        # Get initial values for rolling window
        # Get last 7 actual flux values (convert from log if stored as log)
        initial_values = current_features[f'{model_type}_lag_1'].tail(7).values
        if model_type == 'flux':
            initial_values = 10**initial_values  # uncomment if needed
        rolling_window = deque(initial_values, maxlen=7)   

        # Now if we want to forecast, we need to follow the steps: prediction → rebuild features → prediction
        for step in range(N_future):

            # ---------------------------------------
            # 1. Take last feature row
            # ---------------------------------------
            last_row = current_features.iloc[-1].copy()

             # model expects 2D
            pred_val = model.predict(last_row.values.reshape(1, -1))[0]

            # convert from log space if needed
            if model_type == 'flux':
                pred_val_store = 10**pred_val
            else:
                pred_val_store = pred_val

            future_predictions.append(pred_val_store)

            # Add prediction to rolling window
            rolling_window.append(pred_val)

            # Every step should be 365/5 = 73 days so that we have 5 bins per year giving 10 years of pred
            # next timestamp
            next_time = last_time + (step*73)
            future_times.append(next_time)

            # ---------------------------------------
            # 2. Build NEW feature row
            # ---------------------------------------
            new_row = last_row.copy()

            # ---------------------------------------
            # 3. Update lag columns with prediction
            # ---------------------------------------
            # shift all lags: lag_n <- lag_{n-1}
            for i in reversed(range(len(lag_cols))):
                col = lag_cols[i]
                if i == 0:
                    # lag_1 gets the new prediction
                    new_row[col] = pred_val
                else:
                    # lag_n gets value from lag_{n-1}
                    new_row[col] = last_row[lag_cols[i-1]]

            # ---------------------------------------
            # 4. Recompute rolling features
            # ---------------------------------------

            if "rolling_mean_7" in new_row.index:
                new_row["rolling_mean_7"] = np.mean(rolling_window)

            if "rolling_std_7" in new_row.index:
                new_row["rolling_std_7"] = np.std(rolling_window)

            # ---------------------------------------
            # 5. Append to feature history
            # ---------------------------------------
            new_row.name = next_time
            current_features = pd.concat(
                [current_features, pd.DataFrame([new_row])],
                ignore_index=False
            )

        # -------------------------------------------
        # Done
        # -------------------------------------------
        future_predictions = np.array(future_predictions)
        future_times = np.array(future_times)

        print("Predicted for these times ",future_times)
        np.save(
            f'{save_directory}/seed_{seed}_fl_thresh_{flux_factor_for_alert}_{src}_{model_type}_recursive_predictions.npy',
            [future_times, future_predictions]
        )

        print("\nRecursive forecasting completed.")
        print(f"Generated {len(future_predictions)} future predictions.")

        """ 
        predicted_val = model.predict(features_df)
        print(f"\nPredicted Val statistics:")
        print(f"  Min:   {predicted_val.min():.6f}")
        print(f"  Max:   {predicted_val.max():.6f}")
        print(f"  Mean:  {predicted_val.mean():.6f}")
        print(f"  Std:   {predicted_val.std():.6f}")
                
                
        # Load targets
        targets_df = pd.read_csv(f'{feature_directory}/{src}_{model_type}_targets_{id_val}.csv')
        targets_df.set_index('#MJD', inplace=True)
                
                
        # Create results dataframe
        predictions_df = pd.DataFrame({
                    f'{model_type}_predicted': predicted_val
                }, index=features_df.index)
                
        result_df = predictions_df.join(targets_df, how='inner')
        print(result_df)
                
        # Split train/test
        #train_df = result_df.head(leave_number_indices)
        test_df = result_df.tail(-leave_number_indices)
                
        times = test_df.index.values[1:]   
        if model_type=='flux':
            predictions = 10**(test_df[f'{model_type}_predicted'].values[:-1]) 
        else:
            predictions = test_df[f'{model_type}_predicted'].values[:-1] 
            
        np.save(f'{save_directory}/seed_{seed}_fl_thresh_{flux_factor_for_alert}_{src}_{model_type}_predictions.npy',[times,predictions])
        """    
            
    except Exception as e:
        print(f"ERROR processing {src} ({model_type}): {str(e)}")
        import traceback
        traceback.print_exc()
        return



