## Python script for weighted HEALPix pixelisation
## Assumes data in the format coord1, coord2, value
## supposed to produce pixels with average of "value"

from math import *
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
import os
import sys

##Initialisation

argc = len(sys.argv)
if (argc != 4):
   print "error: call with 'input.file' 'output.file (no extension)' NSide"
   sys.exit()

filein = sys.argv[1]
fileout = sys.argv[2]
nSide = int(sys.argv[3])			##Resolution parameter		

nPix = 12*(nSide)**2 			##Do not change this - always true

fp = open(filein)

pixelSum    = [0]*nPix
pixelWei    = [0]*nPix
pixelWeiSum = [0]*nPix

for n, line in enumerate(fp):
	if (n == 0):
		coord1 = str(line.split()[1])
		coord2 = str(line.split()[2])
		value  = str(line.split()[3])
	elif (n > 0):
		#print 'n = ',n
		l = float(line.split()[0]) # just name for coord1
		b = float(line.split()[1]) # just name for coord2
		wei = float(line.split()[2]) # value to weight counts with

		# print l, b
		theta = (-1.*b + 90.)/180.*pi
		phi = l/180.*pi
		pixelNum = hp.pixelfunc.ang2pix(nSide, theta, phi, nest=False)
		pixelSum[pixelNum] += 1
		pixelWei[pixelNum] = pixelWei[pixelNum]+wei

fp.close()


for i in range(nPix):
	if (pixelSum[i] != 0):
		pixelWeiSum[i] = pixelWei[i]/pixelSum[i]
	else:
		pixelWeiSum[i] = -99

pixelMap = np.array(pixelWeiSum)

# writing to a fits file

fileoutF = fileout + '%s.fits' % nSide
plikMapy = open(fileoutF,'w')
hp.fitsfunc.write_map(plikMapy, pixelMap, nest=False, fits_IDL=True, coord='G')


##Displaying the new density map

hp.mollview(pixelMap, nest = False, min = None, max = None, xsize = 1600, unit = 'average value per pixel')
plt.show()										##Healpy doesn't do Aitoff projections - Mollweide is the closest they've got

##Writing the pixel data to an ASCII file

fileoutN = fileout + '%s.dat' % nSide

fp = open(fileoutN,'w')

fp.write('# pixels in %s,%s coordinates for quantity %s \n# ring scheme \n# nSide %5s \n# pixelNo \t counts/pix \t weight_sum \t avg%s \t  %s \t %s \n' % (coord1,coord2,value,nSide,value, coord1,coord2))

for i in range(nPix):
	coords = hp.pixelfunc.pix2ang(nSide, i, nest=False)
	be = 90.-180.*coords[0]/pi
	el = 180.*coords[1]/pi
	fp.write('%8s \t %8s \t %8s \t %8s \t %8s \t %8s\n' % (i,pixelSum[i],pixelWei[i],pixelMap[i],el,be))

fp.close()