import pyfits
import numpy as np
from matplotlib import pylab as plt
import scipy.optimize as optimization
from  matplotlib.patches import Ellipse

plt.style.use('classic')

#reading data
hdulist = pyfits.open('FQ387N_0.7_0.03_cut_sci_counts.fits')
hdulist2 = pyfits.open('FQ387N_twocomponent_gaussian_01.fits')
hdulist3 = pyfits.open('FQ387N_twocomponent_sersic_10.fits')
hdulist4 = pyfits.open('fq387n_psf.fits')
hdulist5 = pyfits.open('FQ387N_0.7_0.03_weight_tryrms_cut.fits')

#converting data to the correct units
img = hdulist[0].data[210:290,210:290]*34/(12*1429)*2/0.78/0.66*10**(-17)
img2 = hdulist2[2].data[35:65,30:70]*34/(12*1429)*2/0.78/0.66*10**(-17)
img3 = hdulist3[2].data[35:65,30:70]*34/(12*1429)*2/0.78/0.66*10**(-17)
imgpsf = hdulist4[0].data*34*2/0.78/0.66*10**(-17)*0.12    #1.56 for gain 0.12 for scale factor

imgwht = hdulist5[0].data[235:265,230:270]
imgwhtmap = imgwht*34*2/0.78/0.66*10**(-17)
res2= hdulist2[3].data[35:65,30:70]*34/(12*1429)*2/0.78/0.66*10**(-17)
res3= hdulist3[3].data[35:65,30:70]*34/(12*1429)*2/0.78/0.66*10**(-17)



#function for drawing circle in the figure
def circle(centerx,centery):
    x=np.linspace(centerx-3,centerx+3,100)

    y1=np.sqrt(3**2-(np.array(x)-centerx)**2)+centery
    y2=-np.sqrt(3**2-(np.array(x)-centerx)**2)+centery

    return x,y1,y2


def surfacebrightness(centerx,centery,radius,data,error):
    '''
    function for calculating the surface brightness
    :param centerx: xcenter for surface brightness calculation
    :param centery: ycenter for surface brightness calculation
    :param radius: upper limit fot the radius
    :param data: input data
    :param error: error calculation
    :return:
    '''

    x = np.arange(centerx-radius,centerx+radius)
    y = np.arange(centery-radius,centery+radius)
    xx, yy = np.meshgrid(x, y, sparse=False)
    r = np.sqrt((np.array(xx)-centerx)**2+(np.array(yy)-centery)**2)
    index = []
    radii = []
    imgerror = []

    #calculating the surface brightess profile
    for i in np.arange(0,radius,1):
        xi = xx[np.where((r > i) & (r <= i+1))]
        yi = yy[np.where((r > i) & (r <= i+1))]
        print(len(xi))
        radii.append(i + 1)
        print(i, np.mean(data[xi, yi]))
        index.append(np.mean(data[xi,yi]))
        if error is None:
            imgerror = 0
        else:
            imgerror.append(np.sqrt(np.sum((np.array(error[xi,yi]))**2))/len(error[xi,yi]))


    return radii,index,imgerror

#function for fitting surface brightness profile
def func(x, amp,amp2):
    return amp*(np.exp(-x/amp2))

#labeling ellipse aperture
delta = 45.0  # degrees
e = Ellipse((39, 42), 12.674*2, 11.05*2, 60,color='m',fill=False)






#plotting data
plt.subplots_adjust(left=0.2, bottom=0.8, right=None, top=1.6, wspace=0.8, hspace=0.8)
plt.subplot(6,3,1)

plt.xlabel(r'$\rm{kpc}$',fontsize='12')
plt.ylabel(r'$\rm{kpc}$',fontsize='12')
plt.xticks(np.arange(0, 80, 15),round(0.03*8.27,2)*np.arange(0, 80, 15), fontsize=12)
plt.yticks(np.arange(0, 80, 15),round(0.03*8.27,2)*np.arange(0, 80, 15), fontsize=12)
plt.scatter(40,39,c='r', marker = "*",s=5)
e.set_clip_box(plt.subplot(6,3,1).bbox)
e.set_alpha(0.1)
plt.subplot(6,3,1).add_artist(e)

plt.plot(circle(40,39)[0],circle(40,39)[1],linestyle='--',c='green')
plt.plot(circle(40,39)[0],circle(40,39)[2],linestyle='--',c='green')
plt.title('Original Image',fontsize='10')
plt.imshow(img,cmap='Greys',origin='lower')


plt.show()

plt.subplot(6,3,2)
plt.xlabel(r'$\rm{kpc}$',fontsize='12')
plt.ylabel(r'$\rm{kpc}$',fontsize='12')
plt.xticks(np.arange(0, 40, 10),round(0.03*8.27,2)*np.arange(0, 40, 10), fontsize=12)
plt.yticks(np.arange(0, 30, 5),round(0.03*8.27,2)*np.arange(0, 30, 5), fontsize=12)

plt.title('Contour of the Gaussian Fitting Model',fontsize='10')
cs2 = plt.contour(img2,12,cmap='OrRd')
plt.imshow(img2,cmap='binary',origin='lower')

plt.subplot(6,3,3)
plt.xlabel(r'$\rm{kpc}$',fontsize='12')
plt.ylabel(r'$\rm{kpc}$',fontsize='12')
plt.xticks(np.arange(0, 40, 10),round(0.03*8.27,2)*np.arange(0, 40, 10), fontsize=12)
plt.yticks(np.arange(0, 30, 5),round(0.03*8.27,2)*np.arange(0, 30, 5), fontsize=12)
plt.title('Residual of the Gaussian Fitting Model',fontsize='10')
cs3 = plt.contour(img3,12,cmap='OrRd')
plt.title('Contour of the Sersic Fitting Model',fontsize='10')
plt.imshow(img3,cmap='binary',origin='lower')


plt.subplot(6,3,4)

radius = np.array(surfacebrightness(19,18,6,img,imgwht)[0])*0.03*8.27
flux = np.array(surfacebrightness(19,18,6,img,imgwht)[1])/10**(-18)
flux2 = np.array(surfacebrightness(118,118,6,imgpsf,None)[1])/10**(-18)
error = np.array(surfacebrightness(19,18,6,imgpsf,imgwhtmap)[2])/10**(-18)



plt.errorbar(radius,flux,yerr=error,capsize=2, markersize='5', fmt = '.')

plt.plot(radius,flux2,label = 'PSF')
plt.xlabel(r'$\rm{kpc}$')
plt.ylabel(r'$\rm{10^{-18} \ erg \ s^{-1} \ cm^{-2} \ arcsec^{-2}}$')
plt.title('Circularly Surface Brightness Profile ',fontsize='10')
xinput = np.linspace(radius[0],radius[-1],len(radius))
x0 = ([2,2])
best_vals, covar = optimization.curve_fit(func, np.array(radius), flux, x0, sigma=error)
print (best_vals)

plt.plot(xinput, func(np.array(xinput),best_vals[0],best_vals[1]), c = 'red',label = 'exponential fit model')
plt.legend()


plt.subplot(6,3,5)
plt.xlabel(r'$\rm{kpc}$',fontsize='12')
plt.ylabel(r'$\rm{kpc}$',fontsize='12')
plt.xticks(np.arange(0, 40, 10),round(0.03*8.27,2)*np.arange(0, 40, 10), fontsize=12)
plt.yticks(np.arange(0, 30, 5),round(0.03*8.27,2)*np.arange(0, 30, 5), fontsize=12)
plt.title('Residual of the Gaussian Fitting Model',fontsize='10')
plt.imshow(res2,cmap='Greys',origin='lower')

plt.subplot(6,3,6)
plt.xlabel(r'$\rm{kpc}$',fontsize='12')
plt.ylabel(r'$\rm{kpc}$',fontsize='12')
plt.xticks(np.arange(0, 40, 10),round(0.03*8.27,2)*np.arange(0, 40, 10), fontsize=12)
plt.yticks(np.arange(0, 30, 5),round(0.03*8.27,2)*np.arange(0, 30, 5), fontsize=12)
plt.title('Residual of the Sersic Fitting Model',fontsize='10')
plt.imshow(res3,cmap='Greys',origin='lower')



plt.show()










