#!/usr/bin/python
##!env python3

import matplotlib as mpl
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)



def histogram(samples, **kwargs):
	figureWidth = kwargs.get('figureWidth',18)
	figureHeight = kwargs.get('figureHeight',9.2)
	align = kwargs.get('align','left')
	xTicks = kwargs.get('xTicks',None)
	legend = kwargs.get('legend',None)
	stacked = kwargs.get('stacked',False)
	xTickLabels = kwargs.get('xTickLabels', None)
	xlabel = kwargs.get('xlabel','Bins')
	ylabel = kwargs.get('ylabel','Normalized # of occurences')
	show = kwargs.get('show', True)
	ymax = kwargs.get('ymax', None)
	bbins = kwargs.get('bbins', 10)
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	mainTitle = kwargs.get('mainTitle', ' ')
	xMajorLocator = kwargs.get('xMajorLocator', None)
	xMinorLocator = kwargs.get('xMinorLocator', 5)
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	mainTitleFontSize = kwargs.get('mainTitleFontSize', 14)
	xlabelFontSize  = kwargs.get('xlabelFontSize',12)
	ylabelFontSize  = kwargs.get('xlabelFontSize',12)
	tickLabelFontSize = kwargs.get('tickLabelFontSize',14)
	legendSize = kwargs.get('legendSize',6)
	alpha = kwargs.get('alpha',0.3)
	figHist, barx = plt.subplots(figsize=(figureWidth, figureHeight), dpi= 100, facecolor='w', edgecolor='k')
	figHist.suptitle(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
	try:
		samples[0][0]
		multiSamples = True
	except:
		print('Single data')
		multiSamples = False
	def getWeights(samples):
		if stacked == False:
			weights = np.ones_like(samples)/float(len(samples)) # norm
		elif stacked == True:
			j=[len(i) for i in samples]
			totalLength = sum(j)
			weights = []
			for sampleArray in samples:
				weights.append(np.ones_like(sampleArray)/float(totalLength)) # norm
		return weights
	weightsBool = kwargs.get('weights',True)
	if weightsBool == False:
		weightsHist = None
	if multiSamples == False:
		if weightsBool == True:
			weightsHist = getWeights(samples)
		_, a, _ = barx.hist(samples, bins=bbins, edgecolor='black', linewidth=1.2, weights=weightsHist,stacked=stacked, align=align)#density=True, stacked=True)
	else:
		i = 0
		while i < len(samples):
			if weightsBool == True:
				weightsHist = getWeights(samples[i])
			_, a, _ = barx.hist(samples[i], bins=bbins, edgecolor='black', linewidth=1.2, weights=weightsHist,stacked=stacked, align=align,alpha=alpha)#density=True, stacked=True)
			i = i+1
	print(a)
	if legend is not None:
		barx.legend(legend,prop={'size': legendSize})
#	xmin= a[0]
#	xmax = a[-1]
	xmin = kwargs.get('xmin',a[0])
	xmax = kwargs.get('xmax',a[-1])
#	barx.set_xticks(bbins)
	if ymax is not None:
		barx.set_ylim(0, ymax)
	barx.set_xlim(xmin,xmax)
	barx.tick_params(which = 'major', length=15, width = 1, labelrotation=0, axis='both',labelsize=tickLabelFontSize)
	if xMajorLocator is not None:
		barx.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	barx.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	if yMajorLocator is not None:
		barx.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	if yMinorLocator is not None:
		barx.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	barx.set_xlabel(xlabel,fontsize=xlabelFontSize) # r'Trace Time [${\mu}$s'
	barx.set_ylabel(ylabel,fontsize=ylabelFontSize) 
#	
	if xTicks is not None:
		barx.set_xticks(xTicks)
	if xTickLabels is not None:
		figHist.canvas.draw()
		labels = [item.get_text() for item in barx.get_xticklabels()]
		if len(xTickLabels) == 1: 
			labels[xTickLabels[0][0]]=xTickLabels[0][1]
		else:
			for i, val in enumerate(xTickLabels):
				print(val[0], val[1])
				labels[val[0]]=val[1]
		barx.set_xticklabels(labels)
#
	#	/home/tfodran/samplesH_22ch0/
#	save_str = str(hour)+"_"+str(stN)+"_ch"+str(channel)+".png"
#	save_str = str(stN)+"_ch"+str(channel)+".png"
	left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
	bottom = kwargs.get('bottom', 0.15) # the bottom of the subplots of the figure
	right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
	top = kwargs.get('top', 0.88) # the top of the subplots of the figure
	wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
	hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		#plt.close()
	if show == True:
		plt.show()
	return None





def histogram2D(entriesX, entriesY, **kwargs):
	xTickLabels = kwargs.get('xTickLabels', None)
	xlabel = kwargs.get('xlabel','Bins')
	ylabel = kwargs.get('ylabel','Normalized # of occurences')
	show = kwargs.get('show', True)
	ymax = kwargs.get('ymax', None)
	bbins = kwargs.get('bbins', 10)
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	mainTitle = kwargs.get('mainTitle', ' ')
	xMajorLocator = kwargs.get('xMajorLocator', None)
	xMinorLocator = kwargs.get('xMinorLocator', 5)
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	figHist, barx = plt.subplots(figsize=(18, 9.2), dpi= 100, facecolor='w', edgecolor='k')
	figHist.suptitle(mainTitle, fontsize=12, fontweight="bold")
	weightsX = np.ones_like(entriesX)/float(len(entriesX)) # norm
	weightsY = np.ones_like(entriesY)/float(len(entriesY)) # norm
	_, a, _ = barx.hist2D(entriesX, entriesY, bins=bbins, edgecolor='black', linewidth=1.2, weights=[weightsX, weightY])#density=True, stacked=True)
	print(a)
#	xmin= a[0]
#	xmax = a[-1]
	xmin = kwargs.get('xmin',a[0])
	xmax = kwargs.get('xmax',a[-1])
#	barx.set_xticks(bbins)
	if ymax is not None:
		barx.set_ylim(0, ymax)
	barx.set_xlim(xmin,xmax)
	barx.tick_params(which = 'major', length=15, width = 1, labelrotation=0, axis='x')
	if xMajorLocator is not None:
		barx.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	barx.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	if yMajorLocator is not None:
		barx.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	if yMinorLocator is not None:
		barx.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	barx.set_xlabel(xlabel) # r'Trace Time [${\mu}$s'
	barx.set_ylabel(ylabel) 
#	
	if xTickLabels is not None:
		figHist.canvas.draw()
		labels = [item.get_text() for item in barx.get_xticklabels()]
		if len(xTickLabels) == 1: 
			labels[xTickLabels[0][0]]=xTickLabels[0][1]
		else:
			for i, val in enumerate(xTickLabels):
				print(val[0], val[1])
				labels[val[0]]=val[1]
		barx.set_xticklabels(labels)
#
	#	/home/tfodran/entriesH_22ch0/
#	save_str = str(hour)+"_"+str(stN)+"_ch"+str(channel)+".png"
#	save_str = str(stN)+"_ch"+str(channel)+".png"
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		plt.close()
	if show == True:
		plt.show()
	return None


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def histogram2D1Ds(entriesX, entriesY, **kwargs):
	weightsXY = kwargs.get('weightsXY', True)
	yTicks = kwargs.get('yTicks', None)
	rescaleYaxis = kwargs.get('rescaleYaxis', None)
	xTickLabels = kwargs.get('xTickLabels', None)
	yTickLabels = kwargs.get('yTickLabels', None)
	ylabelX = kwargs.get('ylabelX','y value')
	xlabelY = kwargs.get('xlabelY','y value')
	xlabel = kwargs.get('xlabel','x value')
	ylabel = kwargs.get('ylabel','y value')
	show = kwargs.get('show', True)
	bbins = kwargs.get('bbins', [10, 10])
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	mainTitle = kwargs.get('mainTitle', ' ')
	xMajorLocator = kwargs.get('xMajorLocator', None)
	xMajorLocatorY = kwargs.get('xMajorLocatorY', None)
	xMinorLocator = kwargs.get('xMinorLocator', 5)
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	yminX = kwargs.get('yminX',0)
	ymaxX = kwargs.get('ymaxX',0.75)
	yminY = kwargs.get('yminY',0)
	ymaxY = kwargs.get('ymaxY',0.75)
	left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
	bottom = kwargs.get('bottom', 0.11) # the bottom of the subplots of the figure
	right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
	top = kwargs.get('top', 0.88) # the top of the subplots of the figure
	wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
	hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
#	figHist, barx = plt.subplots(figsize=(18, 9.2), dpi= 100, facecolor='w', edgecolor='k')
#	figHist.suptitle(mainTitle, fontsize=12, fontweight="bold")
	#
#
	#
		#
	weightsX = np.ones_like(entriesX)/float(len(entriesX)) # norm
	weightsY = np.ones_like(entriesY)/float(len(entriesY)) # norm
	entriesXY = np.append(entriesX,entriesY)
	if weightsXY is True:
		weightsXY = np.ones_like(entriesX)/float(len(entriesX)) # norm
	elif weightsXY is False:
		weightsXY = np.ones_like(entriesX)
#	cmap = plt.cm.Reds
	#
	tempArray,_,_,_=plt.hist2d(entriesX,entriesY, bins=bbins, weights=weightsXY)
	plt.close()
	maxElement = np.nanmax(tempArray)
	print(maxElement)
	zmaxDef = np.around(maxElement,2)
#
	#
	fig = plt.figure(num=1, figsize=(17,9), dpi=100, facecolor='w', edgecolor='k')
	spec2 = gridspec.GridSpec(ncols=100, nrows=6, figure=fig)
#
	zstep = kwargs.get('zstep', 0.01)
	zminDefault = zstep
	zmin = kwargs.get('zmin', zminDefault)
	zmax = kwargs.get('zmax', zmaxDef)
	#
	extendZmin = kwargs.get('extendZmin',False)
	#
	bounds = np.arange(0,zmax+zstep,zstep)
	cmapT = plt.get_cmap('Greys')
	cmap = truncate_colormap(cmapT, 0.1, 1)
	cmap.set_under('white')
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	#bounds = np.arange(0,zmax+zstep,zstep)
	print('bounds ', bounds)
#	zticks = kwargs.get('zticks', bounds.size)
	#colors = cmap(np.linspace(1.-(zmax-zmin)/float(zmax), 1, cmap.N))
	#cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors)
	#if extendZmin == True:
	#	cmap.set_under('k')
	#
	aXY = fig.add_subplot(spec2[0:4, 0:80])
#
	fig.suptitle(mainTitle, fontsize=12, fontweight="bold") # or plt.suptitle('Main title') #
	histArray, binsX, binsY, im = aXY.hist2d(entriesX, entriesY, bins=bbins, cmap=cmap, weights=weightsXY, vmin=zmin,vmax=zmax,norm=norm)#density=True, stacked=True) edgecolor='white', linewidth=1.2, 
	print(histArray)
	print('test sum: ',np.nansum(histArray))
	print('test max: ',np.nanmax(histArray))
	print('test min: ',np.nanmin(histArray))
#	fig.colorbar(im)
#	fig.colorbar(im, cax = cbaxes)
	cbaxes = fig.add_axes([0.9, 0.38, 0.02, 0.5]) #[left, bottom, width, height],[0.8, 0.1, 0.03, 0.8]    [0.9, 0.05, 0.02, 0.6]
	if extendZmin == True:
		mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, boundaries=bounds,extend='min',ticks=bounds)#, extend='min')
	else:
		mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, boundaries=bounds,ticks=bounds)
	cbaxes.set_ylabel('Normalised # of events')
#
#
	#
	xmin = kwargs.get('xmin',binsX[0])
	xmax = kwargs.get('xmax',binsX[-1])
	ymin = kwargs.get('ymin',binsY[0])
	ymax = kwargs.get('ymax',binsY[-1])
#	barx.set_xticks(bbins)
	aXY.set_xlim(xmin,xmax)
	aXY.set_ylim(ymin,ymax)
#	a.set_yticks([])
#
	aX = fig.add_subplot(spec2[4:6,0:80])
	_, a, _ = aX.hist(entriesX, bins=bbins[0], edgecolor='black', linewidth=1.2, weights=weightsX)#density=True, stacked=True)
	aX.set_xlim(xmin,xmax)
	aX.set_ylim(yminX,ymaxX) # this is basically ymax on the histogram
	aY = fig.add_subplot(spec2[0:4, 81:98])
#	aY.set_xlabel('ADC')
##
	_, a, _ = aY.hist(entriesY, bins=bbins[1], edgecolor='black', linewidth=1.2, weights=weightsY, orientation=u'horizontal')#density=True, stacked=True)
	aY.set_ylim(ymin,ymax)
	aY.set_xlim(yminY,ymaxY) # this is basically ymax on the histogram
	if xMajorLocatorY is not None:
		aY.xaxis.set_major_locator(MultipleLocator(xMajorLocatorY))
	aXY.set_ylabel(ylabel, fontsize='14')
	aX.set_xlabel(xlabel, fontsize='14')
	#
	#tick locators for all
	if xMajorLocator is not None:
		aXY.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
		aX.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	aXY.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	aX.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	if yMajorLocator is not None:
		aXY.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
		aY.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	if yMinorLocator is not None:
		aXY.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
		aY.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	aXY.set_xticklabels([])
	aXY.set_xticks([])
	aY.set_yticklabels([])
	aY.set_yticks([])
	aX.set_ylabel(ylabelX)
	aY.set_xlabel(xlabelY)
#	print(a)
#	barx.grid(b = None)
#	plt.grid(b = None)
#	plt.axis('off')
#	plt.colorbar()
#	xmin= a[0]
#	xmax = a[-1]
	# xmin = kwargs.get('xmin',binsX[0])
	# xmax = kwargs.get('xmax',binsX[-1])
# #	barx.set_xticks(bbins)
	# if ymax is not None:
		# barx.set_ylim(0, ymax)
	# barx.set_xlim(xmin,xmax)
	# barx.tick_params(which = 'major', length=15, width = 1, labelrotation=0, axis='x')
	# # if xMajorLocator is not None:
		# # barx.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	# # barx.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	# # if yMajorLocator is not None:
		# # barx.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	# # if yMinorLocator is not None:
		# # barx.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	# barx.set_xlabel(xlabel) # r'Trace Time [${\mu}$s'
	# barx.set_ylabel(ylabel) 
# #	
	if yTicks is not None:
		aXY.set_yticks(yTicks)
	fig.canvas.draw()
	if xTickLabels is not None:
		labelsX = [item.get_text() for item in aX.get_xticklabels()]
		if len(xTickLabels) == 1: 
			labelsX[xTickLabels[0][0]]=xTickLabels[0][1]
		else:
			for i, val in enumerate(xTickLabels):
				print(val[0], val[1])
				labelsX[val[0]]=val[1]
		aX.set_xticklabels(labelsX)
	if yTickLabels is not None:
		labelsY = [item.get_text() for item in aXY.get_yticklabels()]
		if len(yTickLabels) == 1: 
			labelsY[yTickLabels[0][0]]=yTickLabels[0][1]
		else:
			for i, val in enumerate(yTickLabels):
				print(val[0], val[1])
				labelsY[val[0]]=val[1]
		aXY.set_yticklabels(labelsY)
	if rescaleYaxis is not None:
		labelsFinalY = [item.get_text() for item in aXY.get_yticklabels()]
		aXY.set_yticklabels(np.asarray(labelsFinalY).astype(int)*rescaleYaxis)
# #
	# #	/home/tfodran/entriesH_22ch0/
# #	save_str = str(hour)+"_"+str(stN)+"_ch"+str(channel)+".png"
# #	save_str = str(stN)+"_ch"+str(channel)+".png"
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		plt.close()
	if show == True:
		plt.show()
	return None



def histogram2D1DsOLD(entriesX, entriesY, **kwargs):
	xTickLabels2D = kwargs.get('xTickLabels2D', None)
	xlabel = kwargs.get('xlabel','x value')
	ylabel = kwargs.get('ylabel','y value')
	show = kwargs.get('show', True)
	bbins = kwargs.get('bbins', [10, 10])
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	mainTitle = kwargs.get('mainTitle', ' ')
	xMajorLocator = kwargs.get('xMajorLocator', None)
	xMajorLocatorY = kwargs.get('xMajorLocatorY', None)
	xMinorLocator = kwargs.get('xMinorLocator', 5)
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	yminX = kwargs.get('yminX',0)
	ymaxX = kwargs.get('ymaxX',0.75)
	yminY = kwargs.get('yminY',0)
	ymaxY = kwargs.get('ymaxY',0.75)
	left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
	bottom = kwargs.get('bottom', 0.11) # the bottom of the subplots of the figure
	right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
	top = kwargs.get('top', 0.88) # the top of the subplots of the figure
	wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
	hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
#	figHist, barx = plt.subplots(figsize=(18, 9.2), dpi= 100, facecolor='w', edgecolor='k')
#	figHist.suptitle(mainTitle, fontsize=12, fontweight="bold")
	
	fig = plt.figure(num=1, figsize=(17,9), dpi=100, facecolor='w', edgecolor='k')
	spec2 = gridspec.GridSpec(ncols=100, nrows=6, figure=fig)
	aXY = fig.add_subplot(spec2[0:4, 0:80])
#
	fig.suptitle(mainTitle, fontsize=12, fontweight="bold") # or plt.suptitle('Main title')
	#
		#
	weightsX = np.ones_like(entriesX)/float(len(entriesX)) # norm
	weightsY = np.ones_like(entriesY)/float(len(entriesY)) # norm
	_, binsX, binsY, im = aXY.hist2d(entriesX, entriesY, bins=bbins, cmap=plt.cm.Reds)#, weights=[weightsX, weightsY])#density=True, stacked=True) edgecolor='white', linewidth=1.2, 
#	fig.colorbar(im)
	cbaxes = fig.add_axes([0.9, 0.38, 0.02, 0.5]) #[left, bottom, width, height],[0.8, 0.1, 0.03, 0.8]    [0.9, 0.05, 0.02, 0.6]
	fig.colorbar(im, cax = cbaxes) 
	#
	xmin = kwargs.get('xmin',binsX[0])
	xmax = kwargs.get('xmax',binsX[-1])
	ymin = kwargs.get('ymin',binsY[0])
	ymax = kwargs.get('ymax',binsY[-1])
#	barx.set_xticks(bbins)
	aXY.set_xlim(xmin,xmax)
	aXY.set_ylim(ymin,ymax)
#	a.set_yticks([])
#
	aX = fig.add_subplot(spec2[4:6,0:80])
	_, a, _ = aX.hist(entriesX, bins=bbins[0], edgecolor='black', linewidth=1.2, weights=weightsX)#density=True, stacked=True)
	aX.set_xlim(xmin,xmax)
	aX.set_ylim(yminX,ymaxX) # this is basically ymax on the histogram
	aY = fig.add_subplot(spec2[0:4, 81:98])
#	aY.set_xlabel('ADC')
##
	_, a, _ = aY.hist(entriesY, bins=bbins[1], edgecolor='black', linewidth=1.2, weights=weightsY, orientation=u'horizontal')#density=True, stacked=True)
	aY.set_ylim(ymin,ymax)
	aY.set_xlim(yminY,ymaxY) # this is basically ymax on the histogram
	if xMajorLocatorY is not None:
		aY.xaxis.set_major_locator(MultipleLocator(xMajorLocatorY))
	aXY.set_ylabel(ylabel, fontsize='14')
	aX.set_xlabel(xlabel, fontsize='14')
	#
	#tick locators for all
	if xMajorLocator is not None:
		aXY.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
		aX.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	aXY.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	aX.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	if yMajorLocator is not None:
		aXY.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
		aY.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	if yMinorLocator is not None:
		aXY.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
		aY.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	aXY.set_xticklabels([])
	aXY.set_xticks([])
	aY.set_yticklabels([])
	aY.set_yticks([])
#	print(a)
#	barx.grid(b = None)
#	plt.grid(b = None)
#	plt.axis('off')
#	plt.colorbar()
#	xmin= a[0]
#	xmax = a[-1]
	# xmin = kwargs.get('xmin',binsX[0])
	# xmax = kwargs.get('xmax',binsX[-1])
# #	barx.set_xticks(bbins)
	# if ymax is not None:
		# barx.set_ylim(0, ymax)
	# barx.set_xlim(xmin,xmax)
	# barx.tick_params(which = 'major', length=15, width = 1, labelrotation=0, axis='x')
	# # if xMajorLocator is not None:
		# # barx.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
	# # barx.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
	# # if yMajorLocator is not None:
		# # barx.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	# # if yMinorLocator is not None:
		# # barx.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	# barx.set_xlabel(xlabel) # r'Trace Time [${\mu}$s'
	# barx.set_ylabel(ylabel) 
# #	
	# if xTickLabels is not None:
		# figHist.canvas.draw()
		# labels = [item.get_text() for item in barx.get_xticklabels()]
		# if len(xTickLabels) == 1: 
			# labels[xTickLabels[0][0]]=xTickLabels[0][1]
		# else:
			# for i, val in enumerate(xTickLabels):
				# print(val[0], val[1])
				# labels[val[0]]=val[1]
		# barx.set_xticklabels(labels)
# #
	# #	/home/tfodran/entriesH_22ch0/
# #	save_str = str(hour)+"_"+str(stN)+"_ch"+str(channel)+".png"
# #	save_str = str(stN)+"_ch"+str(channel)+".png"
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		plt.close()
	if show == True:
		plt.show()
	return None


def date_plot(xy, **kwargs):
	totalPlots = len(xy)
	plotDb = kwargs.get('plotDb', False)
	show = kwargs.get('show', True)
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	xerr = kwargs.get('xerr', [None]*totalPlots)
	yerr = kwargs.get('yerr', [None]*totalPlots)
	xlabel = kwargs.get('xlabel', ' ')
	ylabel = kwargs.get('ylabel', ' ')
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	xmin = kwargs.get('xmin', None)
	ymin = kwargs.get('ymin', None)
	xmax = kwargs.get('xmax', None)
	ymax = kwargs.get('ymax', None)
	title = kwargs.get('title', ' ')
	daystack = kwargs.get('daystack', None)
	fulldate = kwargs.get('fulldate', None)
	hline = kwargs.get('hline', None)
	pstyle = kwargs.get('pstyle', ['-']*totalPlots)
	markersize = kwargs.get('markersize', ['2']*totalPlots)
	legend = kwargs.get('legend', None)
	errorBars = kwargs.get('errorBars', False)
	left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
	bottom = kwargs.get('bottom', 0.11) # the bottom of the subplots of the figure
	right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
	top = kwargs.get('top', 0.88) # the top of the subplots of the figure
	wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
	hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
	#
	fig, ax = plt.subplots(figsize=(18, 9.2), dpi= 100, facecolor='w', edgecolor='k')
	#
	ax.xaxis_date()
	if fulldate is None:
		myFmt = mdates.DateFormatter('%H:%M:%S')
	else:
		myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
	#myFmt = mdates.DateFormatter('%H:%M:%S %d %b %Y')
	ax.xaxis.set_major_formatter(myFmt)
#	Xt = xy[0][0].plot_date
	if len(xy) == 1:
		xy.append(['Empty'])
	for i in np.arange(0,len(xy), 1):
		if xy[i][0] != 'Empty':
			if plotDb==True:
				xy[i][1] = 10*np.log10(xy[i][1]/np.min(xy[i][1]))
			ax.errorbar(xy[i][0].plot_date,xy[i][1], xerr=xerr[i], yerr=yerr[i], fmt=pstyle[i], markersize=markersize[i])
#	if errorBars == True:
#		ax.errorbar(Xt, Yvalue,yerr= yerror, fmt = pstyle, capsize=3)
#	else:
#		ax.plot(Xt, Yvalue,pstyle)#, fr, projection3,'g')
	# if (x2 is not None and y2 is not None):
		# Xt2 = x2.plot_date
		# if errorBars2 == True:
			# ax.errorbar(Xt2, y2 ,yerr= yerror2, fmt = pstyle2, capsize=5)
		# else:
			# ax.plot(Xt2, y2,pstyle2, markersize = markersize2)
	#ax.plot(time1x, yhat1, 'k')
	plt.xticks(rotation=20)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.tick_params(axis='x', which='major', length=10, width=2, labelrotation=20)
	if daystack is not None:
		temp = ax.get_xticks()[0]
		ax.set_xlim(int(temp),int(temp)+1)
	if xmin is not None:
		ax.set_xlim(left = xmin)
	if xmax is not None:
		ax.set_xlim(right = xmax)
	if ymin is not None:
		ax.set_ylim(bottom = ymin)
	if ymax is not None:
		ax.set_ylim(top = ymax)
	if hline is not None:
		if len(hline) == 1:
			ax.axhline(hline, alpha=.5, linestyle='--')
		else:
			for i, val in enumerate(hline):
				ax.axhline(hline[i], alpha=.5, linestyle='--')
	if legend is not None:
		ax.legend(legend)
	if yMajorLocator is not None:
		ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
	if yMinorLocator is not None:
		ax.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
	#ax.set_ylim([4,7])
	#ax.set_xticklabels(a_date.value,rotation ='20')
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		plt.close()
	if show == True:
		plt.show()
	return None



# for multipe subplots you have to set rows and columns, example plot2D([ [x1, y1], [x2, y2]])
# for viewing multiple plots in One plot the syntax is: plot2D([ np.array([x1,x2]).T, np.array([y1,y2]).T])
# i.e. you have to create numpy N array of X-variables and one numpy N array of Y-variable and then put them in list [npX , npY] and then this will be in one plot

def plot2d(xy, **kwargs):
	sharey = kwargs.get('sharey', False)
	show = kwargs.get('show', True)
	save = kwargs.get('save',False)
	savePath = kwargs.get('savePath','./defaultName.png')
	rows = kwargs.get('rows', 1)
	columns = kwargs.get('columns', 1)
	rc = rows*columns
	cs = np.arange(0,columns,1)
	rs = np.arange(0,rows,1)
	for i in (np.linspace(0,rc,rc+1)).astype(int):
		if len(xy)/2 != rc:
			xy.append([[0,1],[0,1]])
	text = kwargs.get('text', [None]*rc)
	textCoor = kwargs.get('textCoor',[[0,1]]*rc)
	textBoxSize = kwargs.get('textBoxSize', [10]*rc)
	xlabel = kwargs.get('xlabel', ['']*rc)
	ylabel = kwargs.get('ylabel', ['']*rc)
	xmin = kwargs.get('xmin', ['']*rc)
	ymin = kwargs.get('ymin', ['']*rc)
	xmax = kwargs.get('xmax', ['']*rc)
	ymax = kwargs.get('ymax', ['']*rc)
	alpha = kwargs.get('alpha', [1]*rc)
	mainTitle = kwargs.get('mainTitle', [''])
	mainTitleFontSize = kwargs.get('mainTitleFontSize', 12)
	subtitles = kwargs.get('subtitles', ['']*rc)
	subtitlesFontSize = kwargs.get('subtitlesFontSize', [12]*rc)
	legend = kwargs.get('legend', ['']*rc)
	title = kwargs.get('title', ' ')
	xMajorLocator = kwargs.get('xMajorLocator', None)
	xMinorLocator = kwargs.get('xMinorLocator', None)
	yMajorLocator = kwargs.get('yMajorLocator', None)
	yMinorLocator = kwargs.get('yMinorLocator', None)
	xhighlight = kwargs.get('xhighlight', [None]*(rows*columns))
	vline = kwargs.get('vline', [None]*(rows*columns))
	hline = kwargs.get('hline', [None]*(rows*columns))
	save = kwargs.get('save', None)
	left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
	bottom = kwargs.get('bottom', 0.11) # the bottom of the subplots of the figure
	right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
	top = kwargs.get('top', 0.88) # the top of the subplots of the figure
	wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
	hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
	xlabelFontsize  = kwargs.get('xlabelFontSize',12)
	ylabelFontsize  = kwargs.get('xlabelFontSize',12)
	legendSize = kwargs.get('legendSize',6)
	legendLocation = kwargs.get('legendLocation','upper right')
	#
	if len(xy) != rows*columns:
		if rows*columns - len(xy) == 1:
			xy = (xy, [[0],[0]])
		else:
			for i in (np.arange(0,(rows*columns-len(xy)),1)).astype(int):
				xy = (xy, [[0],[0]])
	if rows*columns > 1:
		fig, ax = plt.subplots(rows,columns, sharey=sharey, figsize=(18, 9.2), facecolor='w', edgecolor='k')
		for axes in ax.flatten():
			#ax.xaxis.set_tick_params(labelbottom=True)
			#ax.yaxis.set_tick_params(labelleft=True)
			axes.yaxis.set_tick_params(labelleft=True)
		if rows == 1 or columns == 1:
			elem = np.arange(0,rows*columns,1)
			elem= list_ungrouping(elem.tolist())
		else:
			elem = combinations_my(rs,cs)
	if rows*columns == 1:
		fig, ax = plt.subplots(figsize=(18, 9.2), facecolor='w', edgecolor='k')
		elem = ['','']
		xy = (xy, [[0],[0]])
	fig.suptitle(mainTitle,fontsize=mainTitleFontSize)
	for i, val in enumerate(elem):
		xyiLength = len(xy[i][0].shape)
		if xyiLength > 1:
			_, colsxyi = xy[i][0].shape
			fmc = kwargs.get('fmc',['b','r','g','k','c','m','y'])
			for k in np.arange(0, colsxyi,1):
				exec('ax'+val+'.plot((xy[i][0])[:,k],(xy[i][1])[:,k],fmc[k], alpha=alpha[i])')
		else:
			fmcBool = kwargs.get('fmcBool', False)
			if fmcBool == True:
				fmc = kwargs.get('fmc',['b','r','g','k','c','m','y'])
				exec('ax'+val+'.plot(xy[i][0],xy[i][1],fmc[i],alpha=alpha[i])')
			elif fmcBool == False:
				exec('ax'+val+'.plot(xy[i][0],xy[i][1],alpha=alpha[i])')
		if text[i] is not None:
			exec('ax'+val+'.text(textCoor[i][0], textCoor[i][1], text[i], size=textBoxSize[i], rotation=0., ha="left", va="top", bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ),transform=ax'+val+'.transAxes)')
		exec('ax'+val+'.set_xlabel(xlabel[i],fontsize=xlabelFontsize)')
		exec('ax'+val+'.set_ylabel(ylabel[i],fontsize=ylabelFontsize)')
		if subtitles is not None:
			exec('ax'+val+'.set_title(subtitles[i],fontsize=subtitlesFontSize[i])')
		if xMajorLocator is not None:
			exec('ax'+val+'.xaxis.set_major_locator(MultipleLocator(xMajorLocator[i]))')
		if xMinorLocator is not None:
			exec('ax'+val+'.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator[i]))')
		if yMajorLocator is not None:
			exec('ax'+val+'.yaxis.set_major_locator(MultipleLocator(yMajorLocator[i]))')
		if yMinorLocator is not None:
			exec('ax'+val+'.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator[i]))')
		if xhighlight[i] is not None:
				exec('ax'+val+".axvspan(xhighlight[i][0], xhighlight[i][1], ymin=0, ymax=1,color='gray')")
		if vline[i] is not None:
			if (len(vline[i]))> 1:
				for vi in vline[i]:
					exec('ax'+val+".axvline(vi,alpha=.5,linestyle='--',color='red')")
			else:
				exec('ax'+val+".axvline(vline[i],alpha=.5,linestyle='--',color='red')")
		if hline[i] is not None:
			if (len(hline[i]))> 1:
				for vi in hline[i]:
					exec('ax'+val+".axhline(vi,alpha=.5,linestyle='--',color='red')")
			else:
				exec('ax'+val+".axhline(hline[i],alpha=.5,linestyle='--',color='red')")
		if xmin[i] != '' :
			exec('ax'+val+'.set_xlim(left = xmin[i])')
		if xmax[i] != '' :
			exec('ax'+val+'.set_xlim(right = xmax[i])')
		if ymin[i] != '' :
			exec('ax'+val+'.set_ylim(bottom = ymin[i])')
		if ymax[i] != '' :
			exec('ax'+val+'.set_ylim(top = ymax[i])')
		if legend[i] != '' :
			exec('ax'+val+".legend(legend[i],loc='upper right',prop={'size': legendSize})")

		if subtitles[i] != '' :
			exec('ax'+val+'.set_title(subtitles[i])')
		if val == '' and i ==0:
			break
	plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
	if save == True:
		plt.savefig(savePath)
		print("Figure has been saved to: ",savePath)
		plt.close()
	if show == True:
		plt.show()
	return None


# xA= np.ones((100)); yA=xA; xA2=xA;yA2=xA;xB=xA;yB=xA;xB2=xA;yB2=xA
# template xy =[[[xA*0,yA],[xA2*0,yA2]],[[xB,yB*2],[xB2,yB2*2]]]
# xy =[[[xA*0,yA]]]
# xy[subplot][1st graph][x or y]


def combinations_my(array1, array2):
    comb = []
    for i1, val1 in enumerate(array1):
        for i2, val2 in enumerate(array2):
            comb.append('['+str(val1)+','+str(val2)+']')
    return comb
#
def list_ungrouping(llist):
    new_list= []
    for i, val in enumerate(llist):
        new_list.append('['+str(val)+']')
    return new_list

def plot2dnew(xy, **kwargs):
    #
    figureWidth = kwargs.get('figureWidth',18)
    figureHeight = kwargs.get('figureHeight',9.2)
    sharey = kwargs.get('sharey', False)
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    rows = kwargs.get('rows', 1)
    columns = kwargs.get('columns', 1)
    rc = rows*columns
    cs = np.arange(0,columns,1)
    rs = np.arange(0,rows,1)
    text = kwargs.get('text', [None]*rc)
    textCoor = kwargs.get('textCoor',[[0,1]]*rc)
    textBoxSize = kwargs.get('textBoxSize', [10]*rc)
    xlabel = kwargs.get('xlabel', ['']*rc)
    ylabel = kwargs.get('ylabel', ['']*rc)
    xmin = kwargs.get('xmin', ['']*rc)
    ymin = kwargs.get('ymin', ['']*rc)
    xmax = kwargs.get('xmax', ['']*rc)
    ymax = kwargs.get('ymax', ['']*rc)
    datePlot = kwargs.get('datePlot',[None]*rc)
    alpha = kwargs.get('alpha', [1]*rc)
    mainTitle = kwargs.get('mainTitle', '')
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 12)
    subtitles = kwargs.get('subtitles', ['']*rc)
    subtitlesFontSize = kwargs.get('subtitlesFontSize', [12]*rc)
    legend = kwargs.get('legend', ['']*rc)
    legendLocation = kwargs.get('legendLocation','upper right')
    legendSize = kwargs.get('legendSize',6)
    title = kwargs.get('title', ' ')
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xtickRotation = kwargs.get('xtickRotation', [0]*rc)
    xtickLength = kwargs.get('xtickLength',[6]*rc)
    xtickWidth = kwargs.get('xtickWidth',[1]*rc)
    xhighlight = kwargs.get('xhighlight', [None]*(rows*columns))
    vline = kwargs.get('vline', [None]*(rows*columns))
    hline = kwargs.get('hline', [None]*(rows*columns))
    save = kwargs.get('save', None)
    left = kwargs.get('left', 0.12) # the left side of the subplots of the figure
    bottom = kwargs.get('bottom', 0.11) # the bottom of the subplots of the figure
    right = kwargs.get('right', 0.9) # the right side of the subplots of the figure
    top = kwargs.get('top', 0.88) # the top of the subplots of the figure
    wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
    hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
    xlabelFontsize  = kwargs.get('xlabelFontSize',12)
    ylabelFontsize  = kwargs.get('ylabelFontSize',12)
    tickLabelFontSize = kwargs.get('tickLabelFontSize',14)
    #
    fmc = kwargs.get('fmc',['b','r','g','k','c','m','y'])
    alpha = kwargs.get('alpha', np.ones(1000))
    if len(xy) == 1:
        print('ok')
        elem = ['']
        fig, ax = plt.subplots(figsize=(figureWidth, figureHeight), facecolor='w', edgecolor='k')
        if datePlot[0] is not None:
            print('date x-axis is on')
            ax.xaxis_date()
            #myFmt = mdates.DateFormatter('%H:%M:%S')
            myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            #myFmt = mdates.DateFormatter('%H:%M:%S %d %b %Y')
            ax.xaxis.set_major_formatter(myFmt)
        for k in range(len(xy[0])):
            print(k)
            ax.plot(xy[0][k][0],xy[0][k][1],fmc[k], alpha=alpha[k])
    elif len(xy) != 1:
        fig, ax = plt.subplots(rows,columns, sharey=sharey, figsize=(figureWidth, figureHeight), facecolor='w', edgecolor='k')
        for axes in ax.flatten():
            axes.yaxis.set_tick_params(labelleft=True)
        if rows == 1 or columns == 1:
            elem = np.arange(0,rows*columns,1)
            elem= list_ungrouping(elem.tolist())
        else:
            elem = combinations_my(rs,cs)
        print(elem)
    for i, val in enumerate(elem):
        print(i)
        if datePlot[i] is not None:
            print('date x-axis is on')
            exec('ax'+val+'.xaxis_date()')
            #myFmt = mdates.DateFormatter('%H:%M:%S')
            myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
            #myFmt = mdates.DateFormatter('%H:%M:%S %d %b %Y')
            exec('ax'+val+'.xaxis.set_major_formatter(myFmt)')
        if len(xy) != 1:
            for k in range(len(xy[i])):
                exec('ax'+val+'.plot(xy[i][k][0],xy[i][k][1],fmc[k], alpha=alpha[k])')
        if text[i] is not None:
            exec('ax'+val+'.text(textCoor[i][0], textCoor[i][1], text[i], size=textBoxSize[i], rotation=0., ha="left", va="top", bbox=dict(boxstyle="square", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ),transform=ax'+val+'.transAxes)')
        exec('ax'+val+'.set_xlabel(xlabel[i],fontsize=xlabelFontsize)')
        exec('ax'+val+'.set_ylabel(ylabel[i],fontsize=ylabelFontsize)')
        if subtitles is not None:
            exec('ax'+val+'.set_title(subtitles[i],fontsize=subtitlesFontSize[i])')
        if xMajorLocator is not None:
            exec('ax'+val+'.xaxis.set_major_locator(MultipleLocator(xMajorLocator[i]))')
        if xMinorLocator is not None:
            exec('ax'+val+'.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator[i]))')
        if yMajorLocator is not None:
            exec('ax'+val+'.yaxis.set_major_locator(MultipleLocator(yMajorLocator[i]))')
        if yMinorLocator is not None:
            exec('ax'+val+'.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator[i]))')
        exec('ax'+val+".xaxis.set_tick_params(rotation=xtickRotation[i],which='major', length=xtickLength[i], width=xtickWidth[i])")
        exec('ax'+val+".tick_params(axis='both', which='major', labelsize=tickLabelFontSize)")
        if xhighlight[i] is not None:
                exec('ax'+val+".axvspan(xhighlight[i][0], xhighlight[i][1], ymin=0, ymax=1,color='gray')")
        if vline[i] is not None:
            if (len(vline[i]))> 1:
                for vi in vline[i]:
                    exec('ax'+val+".axvline(vi,alpha=.5,linestyle='--',color='red')")
            else:
                exec('ax'+val+".axvline(vline[i],alpha=.5,linestyle='--',color='red')")
        if hline[i] is not None:
            if (len(hline[i]))> 1:
                for vi in hline[i]:
                    exec('ax'+val+".axhline(vi,alpha=.5,linestyle='--',color='red')")
            else:
                exec('ax'+val+".axhline(hline[i],alpha=.5,linestyle='--',color='red')")
        if xmin[i] != '' :
            exec('ax'+val+'.set_xlim(left = xmin[i])')
        if xmax[i] != '' :
            exec('ax'+val+'.set_xlim(right = xmax[i])')
        if ymin[i] != '' :
            exec('ax'+val+'.set_ylim(bottom = ymin[i])')
        if ymax[i] != '' :
            exec('ax'+val+'.set_ylim(top = ymax[i])')
        if legend[i] != '' :
            exec('ax'+val+".legend(legend[i],loc=legendLocation,prop={'size': legendSize})")
#
  #      if subtitles[i] != '' :
  #          exec('ax'+val+'.set_title(subtitles[i])')
        if val == '' and i ==0:
            break
    fig.suptitle(mainTitle,fontsize=mainTitleFontSize)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if save == True:
        plt.savefig(savePath)
        print("Figure has been saved to: ",savePath)
        plt.close()
    if show == True:
        plt.show()
    return None


def combinations_my(array1, array2):
    comb = []
    for i1, val1 in enumerate(array1):
        for i2, val2 in enumerate(array2):
            comb.append('['+str(val1)+','+str(val2)+']')
    return comb

def list_ungrouping(llist):
    new_list= []
    for i, val in enumerate(llist):
        new_list.append('['+str(val)+']')
    return new_list


def plot3d(X,Y,Z, **kwargs):
#
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
#
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    cbarLabel = kwargs.get('cbarLabel', None)
    xmin = kwargs.get('xmin', None)
    ymin = kwargs.get('ymin', None)
    zmin = kwargs.get('zmin', None)
    xmax = kwargs.get('xmax', None)
    ymax = kwargs.get('ymax', None)
    zmax = kwargs.get('zmax', None)
    zstep = kwargs.get('zstep', 10)
    mainTitle = kwargs.get('mainTitle', None)
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 16)
    legend = kwargs.get('legend', None)
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xlabelFontsize  = kwargs.get('xlabelFontsize',14)
    ylabelFontsize  = kwargs.get('xlabelFontsize',14)
    cbarLabelFontsize = kwargs.get('cbarLabelFontsize',14)
#
    fig, ax = plt.subplots(figsize=(16.5,9), dpi=100)
    #
    ax.set_title(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
    #
    #colormap
    #cmapT = plt.get_cmap('Greys')
    cmapT = plt.get_cmap('jet')
    lowerColorBarCut = kwargs.get('lowerColorBarCut',0.1)
    upperColorBarCut = kwargs.get('upperColorBarCut', 1)
    cmap = truncate_colormap(cmapT, lowerColorBarCut, upperColorBarCut)
#    cmap.set_under('white')
    cmap.set_under(cmapT(0))
    bounds = np.arange(zmin,zmax+zstep,zstep)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#    cmap = plt.get_cmap('jet') #gist_rainbow  PiYG' gist_ncar
    #
    min_plot = 0 #ampmin*4
    #ampmax//(20)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap,vmin = zmin ,vmax = zmax, norm=norm) #ch0--500 ch1--8
    #
    #
    colorBarAxes = fig.add_axes([0.9, 0.129, 0.03, 0.8]) #[left, bottom, width, height]
    cbarTicks = kwargs.get('cbarTicks',None)
    if cbarTicks is None:
        cbarTicks = bounds
    mpl.colorbar.ColorbarBase(colorBarAxes, cmap=cmap, boundaries=bounds,ticks=cbarTicks, extend='both')#extend='min',
    colorBarAxes.set_ylabel(cbarLabel, fontsize=cbarLabelFontsize)
    #cbar = plt.colorbar(im)
    #cbar.set_label('Amplitude [ADC]', rotation=270, labelpad=10, y=0.5 )
    #
    interval = 4
#
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if xMinorLocator is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
    if yMajorLocator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if yMinorLocator is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
    #
    ax.tick_params(which='major', length=10, width=2)
    ax.tick_params(which='minor', length=4, width=1)
    ax.tick_params(axis='x', which='major', length=15, width=2,labelsize=14)
    ax.tick_params(axis='y', which='major', length=15, width=2,labelsize=14)
    #fig.colorbar(im, ax=ax0)
    ax.set_xlabel(xlabel,fontsize=xlabelFontsize)
    ax.set_ylabel(ylabel,fontsize=ylabelFontsize)
    plt.xticks(rotation=20)
    #
    #
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=True)
    #
    #
    #change the color of the 1st top tick
    #
    left = 0.08  # the left side of the subplots of the figure
    right = 0.89   # the right side of the subplots of the figure
    bottom = 0.13 # the bottom of the subplots of the figure
    top = 0.93    # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
    #              # expressed as a fraction of the average axis width
    hspace = 0.2  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
    #              
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.show()
    # path='/vol/astro6/auger-radiodigitizer/output/'
    # channel = str(channel)
    # savestr = path+"CH"+channel+"_"+update_date+"_"+update_hour+".png"
    # print("Saving figure "+savestr)
    # plt.savefig(savestr)
    return None


# b 	blue
# g 	green
# r 	red
# c 	cyan
# m 	magenta
# y 	yellow
# k 	black


def plot3dnew(X,Y,Z, **kwargs):
#
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    figureWidth = kwargs.get('figureWidth',18)
    figureHeight = kwargs.get('figureHeight',9.2)
#
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    cbarLabel = kwargs.get('cbarLabel', None)
    xmin = kwargs.get('xmin', np.min(X))
    ymin = kwargs.get('ymin', np.min(Y))
    zmin = kwargs.get('zmin', np.min(Z))
    xmax = kwargs.get('xmax', np.max(X))
    ymax = kwargs.get('ymax', np.max(Y))
    zmax = kwargs.get('zmax', np.max(Z))
    Cmin = kwargs.get('Cmin', np.min(Z))
    Cmax = kwargs.get('Cmax', np.max(Z))
    Tmin = kwargs.get('Tmin', Cmin)
    Tmax = kwargs.get('Tmax', Cmax)
    Cstep = kwargs.get('Cstep', (np.max(Cmax)-np.min(Cmin))/100)
    Tstep = kwargs.get('Tstep', 10)
    mainTitle = kwargs.get('mainTitle', None)
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 16)
    legend = kwargs.get('legend', None)
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xlabelFontsize  = kwargs.get('xlabelFontsize',14)
    ylabelFontsize  = kwargs.get('xlabelFontsize',14)
    cbarLabelFontsize = kwargs.get('cbarLabelFontsize',14)
    extend = kwargs.get('extend','neither')
#
    fig, ax = plt.subplots(figsize=((figureWidth, figureHeight)), dpi=100)
    ax.set_title(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
    #
    #colormap
    #cmapT = plt.get_cmap('Greys')
    cmapT = plt.get_cmap('jet')
    print('cmap.T: ',cmapT.N)
    lowerColorBarCut = kwargs.get('lowerColorBarCut',0.1)
    upperColorBarCut = kwargs.get('upperColorBarCut', 1)
    cmap = truncate_colormap(cmapT, lowerColorBarCut, upperColorBarCut)
    cmap.set_under('white')
    cmap.set_under('black')
#    cmap.set_under(cmapT(0))
    bounds = np.arange(Cmin,Cmax+Cstep,Cstep) # minimal, maximal values seen in color scale (range). By python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    print('Bounds: ',bounds)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    print('cmap.N: ',cmap.N)
#    cmap = plt.get_cmap('jet') #gist_rainbow  PiYG' gist_ncar
    #
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, vmin = Cmin ,vmax = Cmax) # by python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    #
    #
    colorBarAxes = fig.add_axes([0.9, 0.129, 0.03, 0.8]) #position of color bar [left, bottom, width, height]
    cbarTicks = kwargs.get('cbarTicks',None)
    if cbarTicks is None:
        cbarTicks = np.arange(Tmin,Tmax+Tstep,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
    print('cbarTicks: ',cbarTicks)
    mpl.colorbar.ColorbarBase(colorBarAxes, cmap=cmap,norm=norm, extend=extend, ticks=cbarTicks )#extend='min', # extend will just point the ends , boundaries=bounds
    colorBarAxes.set_ylabel(cbarLabel, fontsize=cbarLabelFontsize)
    #cbar = plt.colorbar(im)
    #cbar.set_label('Amplitude [ADC]', rotation=270, labelpad=10, y=0.5 )
    #
    interval = 4
#
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if xMinorLocator is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
    if yMajorLocator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if yMinorLocator is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
    #
    ax.tick_params(which='major', length=10, width=2)
    ax.tick_params(which='minor', length=4, width=1)
    ax.tick_params(axis='x', which='major', length=15, width=2,labelsize=14)
    ax.tick_params(axis='y', which='major', length=15, width=2,labelsize=14)
    #fig.colorbar(im, ax=ax0)
    ax.set_xlabel(xlabel,fontsize=xlabelFontsize)
    ax.set_ylabel(ylabel,fontsize=ylabelFontsize)
    plt.xticks(rotation=20)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=True)
    #
    #change the color of the 1st top tick
    #
    left = 0.08  # the left side of the subplots of the figure
    right = 0.89   # the right side of the subplots of the figure
    bottom = 0.13 # the bottom of the subplots of the figure
    top = 0.93    # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
    #              # expressed as a fraction of the average axis width
    hspace = 0.2  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
    #              
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if save == True:
        plt.savefig(savePath)
        print("Figure has been saved to: ",savePath)
        #plt.close()
    if show == True:
        plt.show()
    return None




def plot3dnewV2(X,Y,Z, **kwargs):
#
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    dateFormat = kwargs.get('dateFormat',None)
    figureWidth = kwargs.get('figureWidth',18)
    figureHeight = kwargs.get('figureHeight',9.2)
#
    rotateXlabels = kwargs.get('rotateXlabels',45)
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    cbarLabel = kwargs.get('cbarLabel', None)
    xmin = kwargs.get('xmin', np.min(X))
    ymin = kwargs.get('ymin', np.min(Y))
    zmin = kwargs.get('zmin', np.min(Z))
    xmax = kwargs.get('xmax', np.max(X))
    ymax = kwargs.get('ymax', np.max(Y))
    zmax = kwargs.get('zmax', np.max(Z))
    Cmin = kwargs.get('Cmin', np.min(Z))
    Cmax = kwargs.get('Cmax', np.max(Z))
    Tmin = kwargs.get('Tmin', Cmin)
    Tmax = kwargs.get('Tmax', Cmax)
    Cstep = kwargs.get('Cstep', (np.max(Cmax)-np.min(Cmin))/100)
    Tstep = kwargs.get('Tstep', 9)
    mainTitle = kwargs.get('mainTitle', None)
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 16)
    legend = kwargs.get('legend', None)
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xlabelFontsize  = kwargs.get('xlabelFontsize',14)
    ylabelFontsize  = kwargs.get('xlabelFontsize',14)
    cbarLabelFontsize = kwargs.get('cbarLabelFontsize',14)
    extend = kwargs.get('extend','neither')
#
    fig, ax = plt.subplots(figsize=((figureWidth, figureHeight)))#, dpi=100)
    ax.set_title(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
    if dateFormat != None:
        myFmt = mdates.DateFormatter(dateFormat)
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis_date()
    #
    #colormap
    #cmapT = plt.get_cmap('Greys')
    cmapT = plt.get_cmap('jet')
    print('cmap.T: ',cmapT.N)
    lowerColorBarCut = kwargs.get('lowerColorBarCut',0.1)
    upperColorBarCut = kwargs.get('upperColorBarCut', 1)
    cmap = truncate_colormap(cmapT, lowerColorBarCut, upperColorBarCut)
    cmap.set_under('white')
    cmap.set_under('black')
#    cmap.set_under(cmapT(0))
    bounds = np.arange(Cmin,Cmax+Cstep,Cstep) # minimal, maximal values seen in color scale (range). By python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    #print('Bounds: ',bounds) #debug
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #print('cmap.N: ',cmap.N) #debug
#    cmap = plt.get_cmap('jet') #gist_rainbow  PiYG' gist_ncar
    #
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, vmin = Cmin ,vmax = Cmax) # by python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    #
    #
    cbottom = kwargs.get('cbottom',0.129)
    cheight = kwargs.get('cheight',0.8)
    colorBarAxes = fig.add_axes([0.88, cbottom, 0.03, cheight]) #position of color bar [left, bottom, width, height]
    cbarTicks = kwargs.get('cbarTicks',None)
    if cbarTicks is None:
#        cbarTicks = np.arange(Tmin,Tmax+Tstep,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
        print("tmins: ",Tmin," tmax: ",Tmax, " Tstep: ",Tstep)
        cbarTicks = np.linspace(Tmin,Tmax,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
    print('cbarTicks: ',cbarTicks)
    mpl.colorbar.ColorbarBase(colorBarAxes, cmap=cmap,norm=norm, extend=extend, ticks=cbarTicks )#extend='min', # extend will just point the ends , boundaries=bounds
    colorBarAxes.set_ylabel(cbarLabel, fontsize=cbarLabelFontsize)
    #cbar = plt.colorbar(im)
    #cbar.set_label('Amplitude [ADC]', rotation=270, labelpad=10, y=0.5 )
    #
    interval = 4
#
    if xMajorLocator is not None:
        if dateFormat != None:
            ax.xaxis.set_major_locator(xMajorLocator)
        else:
            ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if xMinorLocator is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
    if yMajorLocator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if yMinorLocator is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
    #
    ax.tick_params(which='major', length=10, width=2)
    ax.tick_params(which='minor', length=4, width=1)
    ax.tick_params(axis='x', which='major', length=15, width=2,labelsize=14)
    ax.tick_params(axis='y', which='major', length=15, width=2,labelsize=14)
    ax.set_ylim(ymin,ymax)
    #fig.colorbar(im, ax=ax0)
    ax.set_xlabel(xlabel,fontsize=xlabelFontsize)
    ax.set_ylabel(ylabel,fontsize=ylabelFontsize)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=True)
    ax.xaxis.set_tick_params(rotation=rotateXlabels,which='major')#, length=xtickLength[i], width=xtickWidth[i])")
    #
    #change the color of the 1st top tick
    #
    left = kwargs.get('left', 0.2) # the left side of the subplots of the figure
    bottom = kwargs.get('bottom', 0.15) # the bottom of the subplots of the figure
    right = kwargs.get('right', 0.85) # the right side of the subplots of the figure
    top = kwargs.get('top', 0.82) # the top of the subplots of the figure
    wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
    hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
    #              
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if save != False:
        plt.savefig(save)
        print("Figure has been saved to: ",save)
        #plt.close()
    if show == True:
        plt.show()
    return None



from matplotlib.colors import ListedColormap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def plot3dnewV3(X,Y,Z, **kwargs):
# #
    close = kwargs.get('close', False)
    dateFormat = kwargs.get('dateFormat',None)
    figureWidth = kwargs.get('figureWidth',18)
    figureHeight = kwargs.get('figureHeight',9.2)
    slices = kwargs.get('slices',False)
    periodicX = kwargs.get('periodicX',False)
#
    if dateFormat is not None:
            rotateXlabels = kwargs.get('rotateXlabels',45)
    else:
        rotateXlabels = kwargs.get('rotateXlabels',0)
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    cbarLabel = kwargs.get('cbarLabel', None)
    xmin = kwargs.get('xmin', np.min(X))
    ymin = kwargs.get('ymin', np.min(Y))
    zmin = kwargs.get('zmin', np.min(Z))
    xmax = kwargs.get('xmax', np.max(X))
    ymax = kwargs.get('ymax', np.max(Y))
    zmax = kwargs.get('zmax', np.max(Z))
    Cmin = kwargs.get('Cmin', np.min(Z))
    Cmax = kwargs.get('Cmax', np.max(Z))
    Tmin = kwargs.get('Tmin', Cmin)
    Tmax = kwargs.get('Tmax', Cmax)
    Cstep = kwargs.get('Cstep', (np.max(Cmax)-np.min(Cmin))/100)
    Tstep = kwargs.get('Tstep', 9)
    mainTitle = kwargs.get('mainTitle', None)
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 16)
    legend = kwargs.get('legend', None)
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xlabelFontsize  = kwargs.get('xlabelFontsize',14)
    ylabelFontsize  = kwargs.get('xlabelFontsize',14)
    cbarLabelFontsize = kwargs.get('cbarLabelFontsize',14)
    extend = kwargs.get('extend','neither')

    fig = plt.figure(num=1, figsize=(figureWidth,figureHeight), dpi=100)#, facecolor='w', edgecolor='k')
    #fig = plt.subplots()#figsize=((figureWidth, figureHeight)))#, dpi=100)
    
    spec = gridspec.GridSpec(ncols=100, nrows=6, figure=fig)
    if slices == True:
        aZ = fig.add_subplot(spec[0:4, 0:80])
    else:
        aZ = fig.add_subplot(spec[0:6, 0:100])
  #  aY = plt.gca() # and reverse
 #   aY.set_xlim(aY.get_xlim()[::-1]) # and reverse
    
    
    aZ.set_title(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
    if dateFormat != None:
        myFmt = mdates.DateFormatter(dateFormat)
        aZ.xaxis.set_major_formatter(myFmt)
        aZ.xaxis_date()
    #
    #colormap
    #cmapT = plt.get_cmap('Greys')
    cmapT = plt.get_cmap('jet')
    print('cmap.T: ',cmapT.N)
    lowerColorBarCut = kwargs.get('lowerColorBarCut',0.1)
    upperColorBarCut = kwargs.get('upperColorBarCut', 1)
    cmap = truncate_colormap(cmapT, lowerColorBarCut, upperColorBarCut)
    cmap.set_under('white')
    cmap.set_under('black')
#    cmap.set_under(cmapT(0))
    bounds = np.arange(Cmin,Cmax+Cstep,Cstep) # minimal, maximal values seen in color scale (range). By python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    #print('Bounds: ',bounds) #debug
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #print('cmap.N: ',cmap.N) #debug
#    cmap = plt.get_cmap('jet') #gist_rainbow  PiYG' gist_ncar
    #
    im = aZ.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, vmin = Cmin ,vmax = Cmax, shading='auto') # by python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    im.set_edgecolor('face')
    #
    #
    cbottom = kwargs.get('cbottom',0.129)
    cheight = kwargs.get('cheight',0.8)
    colorBarAxes = fig.add_axes([0.88, cbottom, 0.03, cheight]) #position of color bar [left, bottom, width, height]
    cbarTicks = kwargs.get('cbarTicks',None)
    if cbarTicks is None:
#        cbarTicks = np.arange(Tmin,Tmax+Tstep,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
        print("tmins: ",Tmin," tmax: ",Tmax, " Tstep: ",Tstep)
        cbarTicks = np.linspace(Tmin,Tmax,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
    print('cbarTicks: ',cbarTicks)
    mpl.colorbar.ColorbarBase(colorBarAxes, cmap=cmap,norm=norm, extend=extend, ticks=cbarTicks )#extend='min', # extend will just point the ends , boundaries=bounds
    colorBarAxes.set_ylabel(cbarLabel, fontsize=cbarLabelFontsize)
    #cbar = plt.colorbar(im)
    #cbar.set_label('Amplitude [ADC]', rotation=270, labelpad=10, y=0.5 )
    #
    interval = 4
#
    if xMajorLocator is not None:
        if dateFormat != None:
            aZ.xaxis.set_major_locator(xMajorLocator)
        else:
            aZ.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if xMinorLocator is not None:
        aZ.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
    if yMajorLocator is not None:
        aZ.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if yMinorLocator is not None:
        aZ.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
    #
    #aZ.tick_params(axis='x', which='major', length=15, width=2,labelsize=14)
    aZ.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=True)
    aZ.tick_params(axis='x',rotation=rotateXlabels,which='major',length=15, width=2,labelsize=14)#, length=xtickLength[i], width=xtickWidth[i])")
    aZ.tick_params(which='minor', length=4, width=1)
    aZ.tick_params(axis='y', which='major', length=15, width=2,labelsize=14)
    aZ.set_ylim(ymin,ymax)
    aZ.set_ylabel(ylabel,fontsize=ylabelFontsize)
    
    if slices == True:
        masked_data = np.ma.masked_array(Z, np.isnan(Z))
        aX = fig.add_subplot(spec[4:6,0:80])
        aY = fig.add_subplot(spec[0:4, 81:98])
        xPlotData = masked_data.mean(axis=0)
        if periodicX == True:
            xPlotData = np.insert(xPlotData,len(xPlotData),xPlotData[0])
        yPlotData = masked_data.mean(axis=1)
        aX.plot(X[1,:],xPlotData, linestyle='-',marker='.')
        aY.plot(yPlotData[::-1],Y[::-1,1], linestyle='-',marker='.') # rotate this one
        aX.set_xlim(aZ.get_xlim())
        aY.set_ylim(aZ.get_ylim())
        aZ.set_xticklabels([])
        aX.set_xlabel(xlabel,fontsize=xlabelFontsize)
        aX.xaxis.set_minor_locator(aZ.xaxis.get_minor_locator())
        aX.xaxis.set_major_locator(aZ.xaxis.get_major_locator())
        aX.tick_params(axis='x',rotation=rotateXlabels,which='major',length=15, width=2,labelsize=14)#, length=xtickLength[i], width=xtickWidth[i])")
        aX.tick_params(which='minor', length=4, width=1)
        aY.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True, bottom=True, top=False, left=False, right=True)
        aX.set_ylabel(cbarLabel)
        aY.set_xlabel(cbarLabel)
        aY.grid(True)
        aX.grid(True)
    else:
        aZ.set_xlabel(xlabel,fontsize=xlabelFontsize)
        
    #aY = plt.gca() # and reverse
    #fig.colorbar(im, aZ=aZ0)


    #
    #change the color of the 1st top tick
    #
    left = kwargs.get('left', 0.1) # the left side of the subplots of the figure old value = 0.2
    bottom = kwargs.get('bottom', 0.15) # the bottom of the subplots of the figure
    right = kwargs.get('right', 0.85) # the right side of the subplots of the figure
    top = kwargs.get('top', 0.93) # the top of the subplots of the figure old value = 0.82
    wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
    hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
    #              
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if save != False:
        try:
            savedir ='/'.join(save.split('/')[:-1])
            os.makedirs(savedir)
        except OSError:
            print ("Creation of the directory failed (already exists?)")
        else:
            print ("Successfully created the directory")
        plt.savefig(save, facecolor='w', transparent=False)
        print("Figure has been saved to: ",save)
    if close == True:
        plt.close()
    if show == True:
        plt.show()
    return None




# fig, axes2d = plt.subplots(nrows=3, ncols=3,
                           # sharex=True, sharey=True,
                           # figsize=(6,6))

# for i, row in enumerate(axes2d):
    # for j, cell in enumerate(row):
        # cell.imshow(np.random.rand(32,32))
        # if i == len(axes2d) - 1:
            # cell.set_xlabel("noise column: {0:d}".format(j + 1))
        # if j == 0:
            # cell.set_ylabel("noise row: {0:d}".format(i + 1))

# plt.tight_layout()

def plot3dnewV4(X,Y,Z, **kwargs): # copied from V3, changes in the subplots
# #
    close = kwargs.get('close', False)
    dateFormat = kwargs.get('dateFormat',None)
    figureWidth = kwargs.get('figureWidth',18)
    figureHeight = kwargs.get('figureHeight',9.2)
    slices = kwargs.get('slices',False)
    periodicX = kwargs.get('periodicX',False)
#
    if dateFormat is not None:
            rotateXlabels = kwargs.get('rotateXlabels',45)
    else:
        rotateXlabels = kwargs.get('rotateXlabels',0)
    show = kwargs.get('show', True)
    save = kwargs.get('save',False)
    savePath = kwargs.get('savePath','./defaultName.png')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    cbarLabel = kwargs.get('cbarLabel', None)
    xmin = kwargs.get('xmin', np.min(X))
    ymin = kwargs.get('ymin', np.min(Y))
    zmin = kwargs.get('zmin', np.min(Z))
    xmax = kwargs.get('xmax', np.max(X))
    ymax = kwargs.get('ymax', np.max(Y))
    zmax = kwargs.get('zmax', np.max(Z))
    Cmin = kwargs.get('Cmin', np.min(Z))
    Cmax = kwargs.get('Cmax', np.max(Z))
    Tmin = kwargs.get('Tmin', Cmin)
    Tmax = kwargs.get('Tmax', Cmax)
    Cstep = kwargs.get('Cstep', (np.max(Cmax)-np.min(Cmin))/100)
    Tstep = kwargs.get('Tstep', 9)
    Ax_ylim = kwargs.get('Ax_ylim', None)
    Ay_xlim = kwargs.get('Ay_xlim', None)
    mainTitle = kwargs.get('mainTitle', None)
    mainTitleFontSize = kwargs.get('mainTitleFontSize', 16)
    legend = kwargs.get('legend', None)
    xMajorLocator = kwargs.get('xMajorLocator', None)
    xMinorLocator = kwargs.get('xMinorLocator', None)
    yMajorLocator = kwargs.get('yMajorLocator', None)
    yMinorLocator = kwargs.get('yMinorLocator', None)
    xlabelFontsize  = kwargs.get('xlabelFontsize',14)
    ylabelFontsize  = kwargs.get('xlabelFontsize',14)
    cbarLabelFontsize = kwargs.get('cbarLabelFontsize',14)
    extend = kwargs.get('extend','neither')

    fig = plt.figure(num=1, figsize=(figureWidth,figureHeight), dpi=100)#, facecolor='w', edgecolor='k')
    #fig = plt.subplots()#figsize=((figureWidth, figureHeight)))#, dpi=100)
    
    spec = gridspec.GridSpec(ncols=100, nrows=6, figure=fig)
    if slices == True:
        aZ = fig.add_subplot(spec[0:4, 0:80])
    else:
        aZ = fig.add_subplot(spec[0:6, 0:100])
  #  aY = plt.gca() # and reverse
 #   aY.set_xlim(aY.get_xlim()[::-1]) # and reverse
    
    
    aZ.set_title(mainTitle, fontsize=mainTitleFontSize, fontweight="bold")
    if dateFormat != None:
        myFmt = mdates.DateFormatter(dateFormat)
        aZ.xaxis.set_major_formatter(myFmt)
        aZ.xaxis_date()
    #
    #colormap
    #cmapT = plt.get_cmap('Greys')
    cmapT = plt.get_cmap('jet')
    print('cmap.T: ',cmapT.N)
    lowerColorBarCut = kwargs.get('lowerColorBarCut',0.1)
    upperColorBarCut = kwargs.get('upperColorBarCut', 1)
    cmap = truncate_colormap(cmapT, lowerColorBarCut, upperColorBarCut)
    cmap.set_under('white')
    cmap.set_under('black')
#    cmap.set_under(cmapT(0))
    bounds = np.arange(Cmin,Cmax+Cstep,Cstep) # minimal, maximal values seen in color scale (range). By python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    #print('Bounds: ',bounds) #debug
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #print('cmap.N: ',cmap.N) #debug
#    cmap = plt.get_cmap('jet') #gist_rainbow  PiYG' gist_ncar
    #
    im = aZ.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, vmin = Cmin ,vmax = Cmax, shading='auto') # by python default Cmin is Zmin, Cmax is Zmax, values < or > are subject of se_under, set_over
    im.set_edgecolor('face')
    #
    #
    cbottom = kwargs.get('cbottom',0.129)
    cheight = kwargs.get('cheight',0.8)
    colorBarAxes = fig.add_axes([0.88, cbottom, 0.03, cheight]) #position of color bar [left, bottom, width, height]
    cbarTicks = kwargs.get('cbarTicks',None)
    if cbarTicks is None:
#        cbarTicks = np.arange(Tmin,Tmax+Tstep,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
        print("tmins: ",Tmin," tmax: ",Tmax, " Tstep: ",Tstep)
        cbarTicks = np.linspace(Tmin,Tmax,Tstep) # by default cbarTicks = bounds, especially when using discrete color scheme
    print('cbarTicks: ',cbarTicks)
    mpl.colorbar.ColorbarBase(colorBarAxes, cmap=cmap,norm=norm, extend=extend, ticks=cbarTicks )#extend='min', # extend will just point the ends , boundaries=bounds
    colorBarAxes.set_ylabel(cbarLabel, fontsize=cbarLabelFontsize)
    #cbar = plt.colorbar(im)
    #cbar.set_label('Amplitude [ADC]', rotation=270, labelpad=10, y=0.5 )
    #
    interval = 4
#
    if xMajorLocator is not None:
        if dateFormat != None:
            aZ.xaxis.set_major_locator(xMajorLocator)
        else:
            aZ.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if xMinorLocator is not None:
        aZ.xaxis.set_minor_locator(AutoMinorLocator(xMinorLocator))
    if yMajorLocator is not None:
        aZ.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if yMinorLocator is not None:
        aZ.yaxis.set_minor_locator(AutoMinorLocator(yMinorLocator))
    #
    #aZ.tick_params(axis='x', which='major', length=15, width=2,labelsize=14)
    aZ.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, bottom=True, top=False, left=True, right=True)
    aZ.tick_params(axis='x',rotation=rotateXlabels,which='major',length=15, width=2,labelsize=14)#, length=xtickLength[i], width=xtickWidth[i])")
    aZ.tick_params(which='minor', length=4, width=1)
    aZ.tick_params(axis='y', which='major', length=15, width=2,labelsize=14)
    aZ.set_ylim(ymin,ymax)
    aZ.set_ylabel(ylabel,fontsize=ylabelFontsize)
    
    if slices == True:
        masked_data = np.ma.masked_array(Z, np.isnan(Z))
        aX = fig.add_subplot(spec[4:6,0:80])
        aY = fig.add_subplot(spec[0:4, 81:98])
        xPlotData = masked_data.mean(axis=0)
        if periodicX == True:
            xPlotData = np.insert(xPlotData,len(xPlotData),xPlotData[0])
        yPlotData = masked_data.mean(axis=1)
        aX.plot(X[1,:],xPlotData, linestyle='-',marker='.')
        aY.plot(yPlotData[::-1],Y[::-1,1], linestyle='-',marker='.') # rotate this one
        aX.set_xlim(aZ.get_xlim())
        aX.set_ylim(Ax_ylim)
        aY.set_ylim(aZ.get_ylim())
        aY.set_xlim(Ay_xlim)
        aZ.set_xticklabels([])
        aX.set_xlabel(xlabel,fontsize=xlabelFontsize)
        aX.xaxis.set_minor_locator(aZ.xaxis.get_minor_locator())
        aX.xaxis.set_major_locator(aZ.xaxis.get_major_locator())
        aX.tick_params(axis='x',rotation=rotateXlabels,which='major',length=15, width=2,labelsize=14)#, length=xtickLength[i], width=xtickWidth[i])")
        aX.tick_params(which='minor', length=4, width=1)
        aY.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True, bottom=True, top=False, left=False, right=True)
        aX.set_ylabel(cbarLabel)
        aY.set_xlabel(cbarLabel)
        aY.grid(True)
        aX.grid(True)
    else:
        aZ.set_xlabel(xlabel,fontsize=xlabelFontsize)
        
    #aY = plt.gca() # and reverse
    #fig.colorbar(im, aZ=aZ0)


    #
    #change the color of the 1st top tick
    #
    left = kwargs.get('left', 0.1) # the left side of the subplots of the figure old value = 0.2
    bottom = kwargs.get('bottom', 0.15) # the bottom of the subplots of the figure
    right = kwargs.get('right', 0.85) # the right side of the subplots of the figure
    top = kwargs.get('top', 0.93) # the top of the subplots of the figure old value = 0.82
    wspace = kwargs.get('wspace', 0.2) # the amount of width reserved for space between subplots, # expressed as a fraction of the average axis width
    hspace = kwargs.get('hspace', 0.2) # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height
    #              
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    if save != False:
        try:
            savedir ='/'.join(save.split('/')[:-1])
            os.makedirs(savedir)
        except OSError:
            print ("Creation of the directory failed (already exists?)")
        else:
            print ("Successfully created the directory")
        plt.savefig(save, facecolor='w', transparent=False)
        print("Figure has been saved to: ",save)
    if close == True:
        plt.close()
    if show == True:
        plt.show()
    return None




# fig, axes2d = plt.subplots(nrows=3, ncols=3,
                           # sharex=True, sharey=True,
                           # figsize=(6,6))

# for i, row in enumerate(axes2d):
    # for j, cell in enumerate(row):
        # cell.imshow(np.random.rand(32,32))
        # if i == len(axes2d) - 1:
            # cell.set_xlabel("noise column: {0:d}".format(j + 1))
        # if j == 0:
            # cell.set_ylabel("noise row: {0:d}".format(i + 1))

# plt.tight_layout()

