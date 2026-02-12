import os,sys
import numpy as np
import pandas as pd
import pyLCR
from astropy.io import fits
import pickle
import datetime

# Codes taken from pyLCR/PlottingTools.py
def computeDate(MET):

    if MET>252460801: MET=MET-1 # 2008 leap second
    if MET>157766400: MET=MET-1 # 2005 leap second
    if MET>362793601: MET=MET-1 # 2012 leap second
    if MET>457401601: MET=MET-1 # 2015 leap second
    if MET>504921601: MET=MET-1 # 2016 leap second
    metdate  = datetime.datetime(2001, 1, 1,0,0,0)
    dt=datetime.timedelta(seconds=MET)
    date=metdate + dt
    yy=date.year
    mm=date.month
    dd=date.day
    hr=date.hour
    mi=date.minute
    ss=date.second
    fff=(float(ss+60.*mi+3600.*hr)/86.4)/1000.0

    return date, fff
    
def computeMJD(MET, returnFraction=True):

    # Get the date and fraction of day for the given MET
    date, fraction = computeDate(MET)

    # Calculate the number of days since January 1, 4713 BC
    # JD = date.toordinal() + 1721424.5
    JD = date.toordinal() + 1721425

    # Calculate the number of days since November 17, 1858
    # MJD = JD - 2400000.5
    MJD = JD - 2400001

    if returnFraction == True:
        MJD = MJD + fraction

    return MJD
    
def get_infor_from_fermi_lc(lightCurve):
    
    # Extract the source name
    source = lightCurve.source

    # Extract likelihood ratio test statistic
    ts = lightCurve.ts

    # Extact the tmin and tmax of the analysis 
    met = lightCurve.met

    # Determine which timebins have detections 
    met_detections = lightCurve.met_detections
    met_upperlimits = lightCurve.met_upperlimits

    # Get the spectral information
    flux_type = lightCurve.flux_type
    index_type = lightCurve.index_type 
    photon_index = lightCurve.photon_index
    photon_index_error = photon_index - lightCurve.photon_index_interval

    # Extract the flux information
    flux = lightCurve.flux
    flux_upper_limit = lightCurve.flux_upper_limits

    # Extracting the flux error and placing it in the proper format
    flux_error = flux - lightCurve.flux_error[:,0]

    # Determine the bin size
    cadence = lightCurve.cadence

    # Quantify the cadence
    if 'daily' in cadence:
        duration = 259_200
    elif 'weekly' in cadence:
        duration = 604_800
    elif 'monthly' in cadence:
        duration = 2_592_000    
        # Get the bin widths
    tmin = met - duration
    tmax = met + duration

    # Create the plot label
    label = flux_type + ' Flux'

    # Get the duration
    dt = tmax-tmin

    # NOTE WE WILL USE MJD FOR EASE!
    
    # Convert dt and x errors into days
    dt = dt / 86400.0
    x_errors = (duration / 2.0) / 86400.0

    # Create lists to store the converted time bins
    MJDs = []
    MJDs_detections = []
    MJDs_upperlimits = []

    # Convert all of the time bins
    for timebin in met:
            MJDs.append(computeMJD(timebin))

    # Convert time bins for the detections
    for timebin in met_detections:
            MJDs_detections.append(computeMJD(timebin))

    # Convert time bins for the nondetections
    for timebin in met_upperlimits:
            MJDs_upperlimits.append(computeMJD(timebin))

    # Vectorize the arrays
    timebins = np.array(MJDs)
    timebins_detections = np.array(MJDs_detections)
    timebins_upperlimits = np.array(MJDs_upperlimits)

    return [timebins_detections,flux],[x_errors,np.transpose(flux_error)],[timebins_upperlimits,flux_upper_limit],[timebins,ts],photon_index