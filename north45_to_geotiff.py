# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:48:49 2020

@author: Kellen
"""

import gdal
import torch
import tb
import numpy as np

fname = f'C:/Users/Kellen/Desktop/D3/Final/pred.npy'
# test = tb.load_ssmi_tb_file(fname)
test = np.load(fname)
test[test==2] = -1

drv = gdal.GetDriverByName("GTiff")

for i in range(len(test)):
    ds = drv.Create(f"fred{i}.tif", 1383, 85, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([-17334193.54, 25067.53, 0, 	-7344784.83, 0, 25067.53])
    ds.SetProjection("+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs")
    
    # for i in range(10):
    #     ds.GetRasterBand(i+1).WriteArray(test[i])
    ds.GetRasterBand(1).WriteArray(test[i])
    ds.FlushCache()
    dst_ds = None
    
