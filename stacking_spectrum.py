import pyfits
import numpy as np

from matplotlib import pylab as plt
from linetools.analysis.voigt import single_voigt_model
import scipy.optimize as optimization
from astropy.modeling.functional_models import Voigt1D
from astropy.stats import sigma_clip
import numpy.ma as ma


f = np.loadtxt('spect_1D_EW.dat')
f2 = np.loadtxt('spect_1D_NS.dat')
f3 = np.loadtxt('J1154_vp.dat')
hdulist1 = pyfits.open('J1154_2D_NS.fits')
hdulist2 = pyfits.open('J1154_2D_EW.fits')
f5 = np.loadtxt('sdss_thpt_387.txt')

data = f
data2 = f2
data3 = f3
data4 = []
data5 = f5

y = np.linspace(0,30,6)
y2 = np.array(y)*10/30.
x = np.linspace(1256,1450,10)

x2 = 3000.+np.array(x)*0.203*3
print (x2)
x3 = (np.array(x2) - 3867.0447)/3867.0447*3*10**5

plt.yticks(y,y2,fontsize=12)
plt.xticks(x,x3,fontsize=12)
plt.contour(hdulist1[0].data[0:30, 1230:1350], 50, levels=[2, 8])
plt.xlabel(r'x pixels')
plt.ylabel('arcsec (")')
plt.title('PA=NS')
plt.imshow(hdulist1[0].data[0:30, 1230:1350])
plt.show()


wave = []
fluxes = []
wave2 = []
fluxes2 = []
wave3 = []
fluxes3 = []
error1 = []
error2 = []
error3 = []
for i in range(0, len(data)):
    wave.append(data[i][0])
    fluxes.append(data[i][1]/(10**(-17)))
    error1.append(data[i][2]/(10**(-17)))
    wave2.append(data2[i][0])
    fluxes2.append(data2[i][1]/(10**(-17)))
    error2.append(data2[i][2]/(10**(-17)))
    wave3.append(data3[i][0])
    error3.append(data3[i][2])
    fluxes3.append(data3[i][1])
    if abs(wave[i] - 3868.096) < 1.2:
        print (i,wave[i])
    if abs(wave[i] - 3852.60) < 0.05:
        print (i,wave[i])


wave5=[]
flux5 = []
for i in range(0, len(data5)):
    wave5.append(data5[i][0])
    flux5.append(data5[i][1]*5.1)

zero_point = 3873.32
ini = 2800
final = 4450

v = (np.array(wave[ini:final])-zero_point)/zero_point*3*10**5
v2 = (np.array(wave2[ini:final])-zero_point)/zero_point*3*10**5
v3 = (np.array(wave3[ini:final])-zero_point)/zero_point*3*10**5
v5 = (np.array(wave5)-zero_point)/zero_point*3*10**5

print (wave[ini])




filtered_data = sigma_clip(fluxes3[ini:final], sigma=3, iters=None, copy=False)



xinput = np.array(np.linspace(wave3[ini],wave3[final],len(wave3[ini:final])))
y = single_voigt_model(21.78,30,2.1853,1216,0.416,6.626*10**8,1.)

k = np.array([20, 30., 2.181, 1216,0.416, 6.2648*10**8.,1.])
def func(x,a,b,c,d):
    in_func = Voigt1D(a,b,c,d)

    return max(in_func(x))-in_func(x)






plt.plot(v,y(xinput),linewidth = 3., label=r"voigt profile for DLA")
plt.plot(v,fluxes3[ini:final],color ="k", linewidth = 0.8, label="original data")
plt.plot(v5[3000:5000],flux5[3000:5000],label="filter transmission")





plt.plot(v, error3[ini:final], color= 'blue', linestyle= "dashed", label = "error average normalized")
plt.xlabel(r'Velocity $(\rm{km \ s^{-1}})$')
plt.ylabel('Flux $10^{-17}$')
plt.legend(loc='upper right')
plt.show()

plt.plot(v, fluxes[ini:final],label = "PA = EW")
plt.plot(v, error1[ini:final],label = "error PA = EW")
plt.plot(v,fluxes2[ini:final],color = 'green',label = "PA = NS")
plt.plot(v, error2[ini:final],color='yellow',label = "error PA = NS")
plt.plot(v,fluxes3[ini:final], color = 'k', label = "average normalized")
plt.plot(v, error3[ini:final],color= 'blue',label = "error average normalized")
plt.axvline(x=0, linewidth=1, color='red')
plt.xlabel(r'Velocity $(\rm{km \ s^{-1}})$')
plt.ylabel('Flux $10^{-17}$')
plt.legend(loc='upper right')

plt.show()


