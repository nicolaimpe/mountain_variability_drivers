#!/bin/bash

# ROI on continental France matching VIIRS resolution (geographic projection EPSG 4326)
RES_X=0.003374578177758
RES_Y=0.0033740359897170007
XMIN=-5.0033746
XMAX=10.003374556489828
YMIN=40.99662600000069
YMAX=51.496626

MNT_FOLDER=./WORK/data/mnt
FOREST_FOLDER=./WORK/data/forest
MSF_FOLDER=./WORK/data/massifs_shapefile
OUT_FOLDER=WORK/data/mnt


# Reprojection of DEM in satellite product projection
gdalwarp -t_srs EPSG:4326 -tr $RES_X $RES_Y -te $XMIN $YMIN $XMAX $YMAX -srcnodata 0 -r lanczos $MNT_FOLDER/DEM_FRANCE_L93_20m_bilinear.tif $OUT_FOLDER/DEM_FRANCE_GEO_375m_lanczos.tif
# Slope and aspect maps (-s 111120 means that the source projection is in lat lon)
gdaldem slope -alg horn  -s 111120 $OUT_FOLDER/DEM_FRANCE_GEO_375m_lanczos.tif $OUT_FOLDER/SLP_FRANCE_GEO_375m_lanczos.tif
gdaldem aspect -alg horn $OUT_FOLDER/DEM_FRANCE_GEO_375m_lanczos.tif $OUT_FOLDER/ASP_FRANCE_GEO_375m_lanczos.tif

# Reprojection of massif mask in satellite product projection
# gdal_rasterize -co COMPRESS=ZSTD -tr 375 375 -a_srs EPSG:2154 -a "code" $MSF_FOLDER/massifs.shp $OUT_FOLDER/MSF_FRANCE_L93_375m.tif
# gdalwarp -t_srs EPSG:4326 -tr $RES_X $RES_Y -te $XMIN $YMIN $XMAX $YMAX -r near $OUT_FOLDER/MSF_FRANCE_L93_375m.tif  $OUT_FOLDER/MSF_FRANCE_GEO_375m.tif 

# Forest mask 
gdalwarp -t_srs EPSG:4326 -tr $RES_X $RES_Y -te $XMIN $YMIN $XMAX $YMAX  -r nearest $FOREST_FOLDER/CORINE_FOREST_FRANCE_EPSG3035_100m.tif $FOREST_FOLDER/CORINE_FOREST_FRANCE_GEO_375m.tif

