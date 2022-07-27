import sqlalchemy as db
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import datetime as dt
import numpy as np
from torch import tensor

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
engine = db.create_engine('postgresql://postgres:postgres@localhost/freezethaw')
conn = engine.connect()
meta = db.MetaData()

def siteList():
    results = conn.execute('SELECT * FROM''"Borehole_sites"')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df


def dateList():
    results = conn.execute('SELECT * FROM''"Dates"')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df

def siteQuery(sites):
    sitetext = f'"Sid" = {sites[0]}'
    if len(sites) > 1:
        for i in range(1, len(sites)):
            sitetext += f'AND "Sid" = {sites[i]}'
    results = conn.execute('SELECT * FROM'f'"Borehole_smooth" where({sitetext})')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df


def dateQuery(dateStart, dateEnd=None, site=None):
    if dateEnd:
        datetext = '"date" between timestamp ' f"'{dateStart}' AND '{dateEnd}'"
    else:
        datetext = '"date" = 'f"'{dateStart}'"
    if site:
        datetext += f' AND "Sid" = {site}'
    results = conn.execute('SELECT * FROM'f'"Borehole_smooth" where({datetext})')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df

def dateQuery2(dateStart, dateEnd=None):
    if dateEnd:
        datetext = 'date between timestamp ' f"'{dateStart}' AND '{dateEnd}'"
    else:
        datetext = 'date = 'f"'{dateStart}'"
    results = conn.execute('SELECT * FROM'f' learning where({datetext}) ORDER BY index')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df

def dateQuery3(dateStart, kwargs=[]):
    datetext = 'date = 'f"'{str(dateStart)}'"
    results = conn.execute('SELECT * FROM'f' learningarrays3 where({datetext})')
    res = []
    res2 = []
    for row in results:
        if 'tb19' not in kwargs:
            res.append(row[1])
            res.append(row[2])
        if 'tb37' not in kwargs:
            res.append(row[3])
            res.append(row[4])
        if 'elev' not in kwargs:
            res.append(row[5])
        if 'day' not in kwargs:
            res.append(row[6])
        if 'snow' not in kwargs:
            res.append(row[7])
        res2.extend(row[8])
    return tensor(res).float().cuda(),tensor(res2).float().cuda()

def dateQueryair(dateStart, kwargs=[]):
    datetext = 'date = 'f"'{str(dateStart)}'"
    results = conn.execute('SELECT * FROM'f' learningarraysair where({datetext})')
    res = []
    res2 = []
    for row in results:
        if 'tb19' not in kwargs:
            res.append(row[1])
            res.append(row[2])
        if 'tb37' not in kwargs:
            res.append(row[3])
            res.append(row[4])
        if 'elev' not in kwargs:
            res.append(row[5])
        if 'day' not in kwargs:
            res.append(row[6])
        if 'snow' not in kwargs:
            res.append(row[7])
        res2.extend(row[8])
    return tensor(res).float().cuda(),tensor(res2).float().cuda()
    # return res,res2
def canakQuery(dateStart, kwargs=[]):
    datetext = 'date = 'f"'{str(dateStart)}'"
    results = conn.execute('SELECT * FROM'f' canak where({datetext})')
    res = []
    res2 = []
    for row in results:
        if 'tb19' not in kwargs:
            res.append(row[1])
            res.append(row[2])
        if 'tb37' not in kwargs:
            res.append(row[3])
            res.append(row[4])
        if 'snow' not in kwargs:
            res.append(row[5])
        if 'land' not in kwargs:
            res.append(row[6])
        res2.extend(row[7])
    return tensor(res).float().cuda(),tensor(res2).float().cuda()

def locQuery(latStart, lonStart,latEnd = None, lonEnd=None):
    if latEnd:
        latString = f'"Borehole_sites"."lat" BETWEEN {latStart} AND {latEnd}'
    else:
        latString = f'"Borehole_sites"."lat" = {latStart}'
    if lonEnd:
        lonString = f'"Borehole_sites"."lon" BETWEEN {lonStart} AND {lonEnd}'
    else:
        lonString = f'"Borehole_sites"."lon" = {lonStart}'
    results = conn.execute(f'SELECT "Borehole_data".* FROM "Borehole_data","Borehole_sites" WHERE("Borehole_data"."Sid" = "Borehole_sites"."Sid" AND {latString} AND {lonString});')
    df = pd.DataFrame(results)
    df.columns = results.keys()
    return df
