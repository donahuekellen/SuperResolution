import numpy as np
import scipy.interpolate as inter
import h5py as h
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon
import datetime
import torch
import ease_grid as eg
import tb


start = datetime.datetime(2015, 5, 5, 6, 0, 0)
end = datetime.datetime(2020, 6, 15, 6, 0, 0)
train_dates = [start + datetime.timedelta(days=i) for i in range((end - start).days)]


lon, lat = eg.v1_get_full_grid_lonlat(eg.ML)
lon[lon<0] += 360
lat = 90-lat

points = [x for x in zip(lon.flatten(),lat.flatten())]
la = 720
lo = 1440

lats = [i/4 for i in range(la)]
lons = [i/4 for i in range(lo)]
alons = [lons[i%lo] for i in range(la*lo)]
alats = [lats[int(i/lo)] for i in range(la*lo)]
apoints = [x for x in zip(alons,alats)]
date = datetime.datetime(2017, 2, 1, 6, 0, 0)
f18 = h.File(
        f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB18GHz_25/2/{date.year}/{date.month:02}/GW1AM2_{date.year}{date.month:02}{date.day:02}_01D_EQMA_L3SGT18LA2220220.h5')
v18 = np.array(f18['Brightness Temperature (V)']).astype(float)
test = inter.NearestNDInterpolator(apoints, v18.flatten())

for date in train_dates:
        try:
                print(date.date())
                temp = torch.load(f'C://Users/Kellen/Desktop/Machinelearning/newdata/{date.date()}.pt')
                f18 = h.File(
                        f'H://AMSR2/ftp.gportal.jaxa.jp/standard/GCOM-W/GCOM-W.AMSR2/L3.TB18GHz_25/2/{date.year}/{date.month:02}/GW1AM2_{date.year}{date.month:02}{date.day:02}_01D_EQMA_L3SGT18LA2220220.h5')
                v18 = np.array(f18['Brightness Temperature (V)']).astype(float)
                h18 = np.array(f18['Brightness Temperature (H)']).astype(float)
                
                test.values = v18.flatten()
                grid = test(points)
                test.values = h18.flatten()
                grid1 = test(points)
                
                grid = grid.reshape(586,1383)
                grid1 = grid1.reshape(586, 1383)
                
                grid = grid/100
                grid[grid >= 400] = 0
                grid[grid <= 0] = 0
                grid[grid > 0] = grid[grid > 0] / 400
                grid1 = grid1 / 100
                grid1[grid1 >= 400] = 0
                grid1[grid1 <= 0] = 0
                grid1[grid1 > 0] = grid1[grid1 > 0] / 400
                temp[2] = torch.tensor(grid)
                temp[3] = torch.tensor(grid1)
                

                torch.save(temp,f'C://Users/Kellen/Desktop/Machinelearning/newdata/{date.date()}.pt',)
                
        except:
                print("missing", date.date())
                missinglist.append(date)
for i in missinglist:
        print(i)

plt.imshow(test)
plt.show()