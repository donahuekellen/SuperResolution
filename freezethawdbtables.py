import sqlalchemy as db
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import DBcheck as db2
import tb
import datetime
import netcdfreader as nc
import glob
import os
import rasterio as rio
import numpy as np
pd.options.mode.chained_assignment = None
engine = db.create_engine('postgresql://postgres:postgres@localhost/freezethaw')
conn = engine.connect()
meta = db.MetaData()




# sites = db.Table('Borehole_sites', meta,
#                 db.Column('Sid', db.Integer(),primary_key=True),
#                 db.Column('lat', db.Float()),
#                 db.Column('lon', db.Float())
#                  )
# dates = db.Table('Dates',meta,
#                 db.Column('date', db.DateTime(),primary_key=True)
#                  )
learningarrays = db.Table('learningarrays', meta,
                db.Column('date', db.DateTime()),
                db.Column('tb19v', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('tb19h', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('tb37v', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('tb37h', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('elev', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('day', db.types.ARRAY(db.Float,dimensions=2)),
                db.Column('temp', db.types.ARRAY(db.Float,dimensions=2)),
                )
#
# smooth = db.Table('Borehole_smooth', meta,
#                 db.Column('Sid', db.Integer(),db.ForeignKey("Borehole_sites.Sid")),
#                 db.Column('date', db.DateTime(),db.ForeignKey("Dates.date")),
#                 db.Column('soilTemp0', db.Float()),
#                 db.Column('soilTemp1', db.Float()),
#                 db.Column('soilTemp2', db.Float()),
#                 db.Column('soilTemp4', db.Float()),
#                 db.Column('soilTemp8', db.Float()),
#                 db.Column('soilTemp20', db.Float()),
#                 db.Column('soilMoist1', db.Float()),
#                 db.Column('soilMoist2', db.Float()),
#                 db.Column('soilMoist4', db.Float()),
#                 db.Column('soilMoist8', db.Float()),
#                 db.Column('soilMoist20', db.Float()),
#                 db.Column('t2m1', db.Float()),
#                 db.Column('snowDepth1', db.Float()),
#                 )
meta.create_all(engine)

root = '/mnt/data01/repr/'
x = pd.read_csv(root + '2017/01/01/reworked.csv',delimiter = ',')

start = datetime.datetime(2015, 1,1,6,0,0)
end = datetime.datetime(2019, 1,1,6,0,0)
date_list = [start+datetime.timedelta(days=i) for i in range((end-start).days)]
print(f'loading training data for {str(start.date())} to {str(end.date())}...')

londict = {}
latdict = {}
lonindex = np.unique(x['lon'])
latindex = np.unique(x['lat'])

for lon in lonindex:
    londict.update({lon: np.where(lonindex == lon)[0][0]})

for lat in latindex:
    latdict.update({lat: np.where(latindex == lat)[0][0]})

test = np.ndarray((len(latindex), len(lonindex)))
test.fill(-1)
tb19v = test.copy()
tb37v = test.copy()
tb19h = test.copy()
tb37h = test.copy()
elev = test.copy()
date = test.copy()
temp = test.copy()
ins = learningarrays.insert()
for d in date_list:
    print(d)
    query = db2.dateQuery2(str(d))
    day = (d - datetime.datetime(d.year - 1, 12, 31)).days/366
    for i in range(len(query)):
        tb19v[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['tb19v'].iloc[i]
        tb37v[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['tb37v'].iloc[i]
        tb19h[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['tb19h'].iloc[i]
        tb37h[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['tb37h'].iloc[i]
        elev[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['elevation'].iloc[i]
        date[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = day
        temp[latdict[x['lat'].iloc[i]]][londict[x['lon'].iloc[i]]] = query['temp'].iloc[i]
    conn.execute(ins,date=d,tb19v=tb19v,tb19h=tb19h,tb37v=tb37v,tb37h=tb37h,elev=elev,day=date,temp=temp)




conn.close()