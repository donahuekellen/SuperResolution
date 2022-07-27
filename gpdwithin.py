import geopandas as gpd

def within(df1,df2,how='inner',op='within',to_crs2=False):
    crs1 = df1.crs
    crs2 = df2.crs
    if to_crs2:
        df1.to_crs(crs2)
    else:
        df2.to_crs(crs1)
    return gpd.sjoin(df1,df2, how=how, op=op)

