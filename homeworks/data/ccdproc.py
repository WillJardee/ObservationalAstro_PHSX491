#!/usr/bin/env python

# CCDPROC code

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
from glob import glob
import time
#import matplotlib
#matplotlib.use('nbagg')

def ccdlist(input=None):
    if input is None: input='*.fits'
    files = glob(input)
    nfiles = len(files)
    dt = np.dtype([('file',np.str,100),('object',np.str,100),('naxis1',int),('naxis2',int),
                      ('imagetyp',np.str,100),('exptime',float),('filter',np.str,100)])
    cat = np.zeros(nfiles,dtype=dt)
    for i,f in enumerate(files):
        base = os.path.basename(f)
        base = base.split('.')[0]
        h = fits.getheader(f)
        cat['file'][i] = f
        cat['object'][i] = h.get('object')
        cat['naxis1'][i] = h.get('naxis1')
        cat['naxis2'][i] = h.get('naxis2')
        cat['imagetyp'][i] = h.get('imagetyp')
        cat['exptime'][i] = h.get('exptime')
        cat['filter'][i] = h.get('filter')
        print(base+'  '+str(cat['naxis1'][i])+'  '+str(cat['naxis2'][i])+'  '+cat['imagetyp'][i]+'  '+str(cat['exptime'][i])+'  '+cat['filter'][i])
    return cat

def fixheader(head):
    """ Update the headers."""
    head2 = head.copy()
    head2['BIASSEC1'] = '[1:41,1:2728]'
    head2['BIASSEC2'] = '[3430:3465,1:2728]'
    head2['TRIMSEC'] = '[42:3429,15:2726]'
    head2['DATASEC'] = '[42:3429,15:2726]'
    head2['RDNOISE'] = (4.5, 'readnoise in e-')
    head2['GAIN'] = (0.15759900212287903, 'Electronic gain in e-/ADU')
    head2['BUNIT'] = 'ADU'
    return head2

    
def overscan(im,head):
    """ This calculate the overscan and subtracts it from the data and then trims off the overscan region"""
    # y = [0:40] and [3429:3464]
    # x = [0:13] and [2726:2727]
    # DATA = [14:2725,41:3428]
    # 2712 x 3388
    nx,ny = im.shape
    o1 = im[:,0:41]
    o2 = im[:,3429:3465]
    o = np.hstack((o1,o2))
    # Take the mean
    mno = np.mean(o,axis=1)
    # Fit line to it
    coef = np.polyfit(np.arange(nx),mno,1)
    fit = np.poly1d(coef)(np.arange(nx))
    # Subtract from entire image
    oim = np.repeat(fit,ny).reshape(nx,ny)
    out = im.astype(float)-oim
    # Trim the overscan
    out = out[14:2726,41:3429]
    # Update header
    nx1, ny1 = out.shape
    head2 = head.copy()
    head2['NAXIS1'] = ny1
    head2['NAXIS2'] = nx1
    head2['BIASSEC1'] = '[1:41,1:2728]'
    head2['BIASSEC2'] = '[3430:3465,1:2728]'
    head2['TRIMSEC'] = '[42:3429,15:2726]'
    head2['OVSNMEAN'] = np.mean(oim)
    head2['TRIM'] = time.ctime()+' Trim is [42:3429,15:2726]'
    head2['OVERSCAN'] = time.ctime()+' Overscan is [1:41,1:2728] and [3430:3465,1:2728], mean '+str(np.mean(oim))
    #head2.add_history('Overscan corrected and trimmed on '+time.ctime())
    return out, head2
    
def masterbias(files,outfile=None,clobber=True):
    """
    Load the bias images.  Overscan correct and trim them.  Then average them.

    Parameters
    ----------
    files : list
        List of bias FITS files.
    outfile : string, optional
        Filename to write the master bias image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master bias image.
    ahead : header dictionary
        The master bias header.

    Example
    -------

    bias, bhead = masterbias(bias_files)

    """

    nfiles = len(files)
    imarr = np.zeros((2712, 3388, nfiles),float)
    for i in range(nfiles):
        print(str(i+1)+' '+files[i])
        im,head = fits.getdata(files[i],0,header=True)
        im2,head2 = ccdproc(im,head)
        imarr[:,:,i] = im2
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    aim = np.mean(imarr,axis=2)
    ahead['NCOMBINE'] = nfiles
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master bias to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead


def overscanzero(im,head,zero):
    """ Overscan subtract, trim, subtract master zero"""
    im2 = overscan(im)
    return im2 - zero

def masterdark(files,zero,outfile=None,clobber=True):
    """
    Load the dark images.  Overscan correct and trim them.  zero subtract.  Then average them.

    Parameters
    ----------
    files : list
        List of dark FITS files.
    outfile : string, optional
        Filename to write the master dark image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master dark image.
    ahead : header dictionary
        The master dark header.

    Example
    -------

    dark, dhead = masterdark(dark_files)

    """

    nfiles = len(files)
    imarr = np.zeros((2712, 3388, nfiles),float)
    for i in range(nfiles):
        print(str(i+1)+' '+files[i])
        im,head = fits.getdata(files[i],0,header=True)
        im2,head2 = ccdproc(im,head,zero)
        imarr[:,:,i] = im2 / np.median(im2)
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    # Take average
    aim = np.mean(imarr,axis=2)
    # Divide by exposure time
    aim /= head['exptime']
    ahead['NCOMBINE'] = nfiles
    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master dark to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)
    
    return aim, ahead

def overscantrimzerodark(im,head,zero,dark):
    """ Overscan subtract, trim, subtract master zero, subtract master dark"""
    # Overscan subtract and trim
    im2 = overscantrim(im)
    # Subtract master bias
    im2 -= zero
    # Subtract master dark scaled to this exposure time
    im2 -= dark*head['exptime']
    return im2

def masterflat(files,zero,dark,outfile=None,clobber=True):
    """
    Load the flat images.  Overscan correct and trim them.  Bias and dark subtract. Then divide by median and average them.

    Parameters
    ----------
    files : list
        List of flat FITS files.
    outfile : string, optional
        Filename to write the master flat image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    aim : numpy image
        The 2D master flat image.
    ahead : header dictionary
        The master flat header.

    Example
    -------

    flat, fhead = masterflat(flat_files)

    """

    nfiles = len(files)
    imarr = np.zeros((2712, 3388, nfiles),float)
    for i in range(nfiles):
        im,head = fits.getdata(files[i],0,header=True)
        print(str(i+1)+' '+files[i]+' '+str(head.get('FILTER')))
        im2,head2 = ccdproc(im,head,zero,dark)
        imarr[:,:,i] = im2 / np.median(im2)
        if i==0: ahead=head2.copy()
        ahead['CMB'+str(i+1)] = files[i]
    aim = np.mean(imarr,axis=2)
    ahead['NCOMBINE'] = nfiles

    # Output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing master flat to '+outfile)
        hdu = fits.PrimaryHDU(aim,ahead).writeto(outfile)

    return aim, ahead

def ccdproc(data,head=None,zero=None,dark=None,flat=None,outfile=None,clobber=True):
    """
    Overscan subtract, trim, subtract master zero, subtract master dark, flat field.

    Parameters
    ----------
    data : list or numpy 2D array
        This can either be a list of image filenames or a 2D image.
    head : header dictionary, optional
        The header if an image is input.
    zero : filename or numpy 2D array, optional
        The master bias.  Either the 2D image or the filename.
    dark : filename or numpy 2D array, optional
        The master dark.  Either the 2D image or the filename.
    flat : filename or numpy 2D array, optional
        The master flat.  Either the 2D image or the filename.
    outfile : string, optional
        Filename to write the processed image to.
    clobber : bool, optional
        If the output file already exists, then overwrite it.

    Returns
    -------
    fim : numpy image
        The 2D processed image.
    fhead : header dictionary
        The header for the processed image.

    Example
    -------

    flat, fhead = ccdproc(files,zero,dark,flat)

    """

    # Filename input
    if type(data) is str:
        if os.path.exists(data):
            im,head = fits.getdata(date,0,header=True)
        else:
            raise ValueError(data+' NOT FOUND')
    # Image input
    else:
        im = data
        if head is None:
            raise ValueError('Header not input')

    # Fix header, if necessary
    if (head.get('TRIMSEC') is None) | (head.get('BIASSEC') is None):
        head = fixheader(head)
        
    # Overscan subtract and trim
    #---------------------------
    if head.get('OVERSCAN') is None:
        fim,fhead = overscan(im,head)
    else:
        print('Already OVERSCAN corrected')
        fim = im.copy()
        fhead = head.copy()
    # Subtract master bias
    #---------------------
    if (zero is not None):
        # Not corrected yet
        if head.get('ZEROCOR') is None:
            # Filename input
            if type(zero) is str:
                if os.path.exists(zero):
                    zeroim,zerohead = fits.getdata(zero,0,header=True)
                else:
                    raise ValueError(zero+' NOT FOUND')
            # Image input
            else:
                zeroim = zero
            # Do the correction
            fim -= zeroim
            fhead['ZEROCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % (np.mean(zeroim),np.std(zeroim))
        # Corrected already
        else:
            print('Already ZERO subtracted')
    # Subtract master dark scaled to this exposure time
    #--------------------------------------------------
    if (dark is not None):
        # Not corrected yet
        if head.get('DARKCOR') is None:
            # Filename input
            if type(dark) is str:
                if os.path.exists(dark):
                    darkim,darkhead = fits.getdata(dark,0,header=True)
                else:
                    raise ValueError(dark+' NOT FOUND')
            # Image input
            else:
                darkim = dark
            # Do the correction
            fim -= darkim*head['exptime']
            fhead['DARKCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % \
                               (np.mean(darkim*head['exptime']),np.std(darkim*head['exptime']))        
        # Corrected already
        else:
            print('Already DARK corrected')
    # Flat field
    if (flat is not None):
        # Not corrected yet
        if head.get('FLATCOR') is None:
            # Filename input
            if type(flat) is str:
                if os.path.exists(flat):
                    flatim,flathead = fits.getdata(flat,0,header=True)
                else:
                    raise ValueError(flat+' NOT FOUND')
            # Image input
            else:
                flatim = flat
            # Do the correction
            fim /= flatim
            fhead['FLATCOR'] = time.ctime()+' mean %6.2f, stdev %6.2f' % (np.mean(flatim),np.std(flatim))
        # Already corrected
        else:
            print('Already FLAT corrected')

    fhead['CCDPROC'] = time.ctime()+' CCD processing done'

    # Write to output file
    if outfile is not None:
        if os.path.exists(outfile):
            if clobber is False:
                raise ValueError(outfile+' already exists and clobber=False')
            else:
                os.remove(outfile)
        print('Writing processed file to '+outfile)
        hdu = fits.PrimaryHDU(fim,fhead).writeto(outfile)
    
    return fim, fhead

