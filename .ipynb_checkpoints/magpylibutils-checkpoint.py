# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from magpylib._lib.mathLib import angleAxisRotation
from magpylib._lib.classes.magnets import Box
from magpylib._lib.classes.sensor import Sensor


# %%
class DiscreteSourceBox(Box):
    def __init__(self, data, bounds_error=None, fill_value=None):
        '''bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used.
        fill_value : number, optional
            If provided, the value to use for points outside of the interpolation domain. If None, values outside the domain are extrapolated.
        '''
        m = np.min(data,axis=0)
        M = np.max(data,axis=0)
        self.dimension = (M-m)[:3]
        self._center = 0.5*(M+m)[:3]
        self.position = self._center
        self.magnetization = (0,0,0)
        self.angle = 0
        self.axis = (0,0,1)
        self.interpFunc = self._interpolate_data(data, bounds_error=bounds_error, fill_value=fill_value)
        self.data = data
    
    def getB(self, pos):
        pos += self._center - self.position
        B = self.interpFunc(pos)
        if self.angle!=0:
            return np.array([angleAxisRotation(b, self.angle, self.axis) for b in B])
        else:
            return B[0] if B.shape[0] == 1 else B
    
    def _interpolate_data(self, data, bounds_error, fill_value):
        '''Define interpolating functions for B field values, values need to be sorted in x,y,z order'''
        x,y,z,Bx,By,Bz = data.T
        nx,ny,nz = len(np.unique(x)), len(np.unique(y)), len(np.unique(z))
        X = np.linspace(np.min(x), np.max(x), nx)
        Y = np.linspace(np.min(y), np.max(y), ny)
        Z = np.linspace(np.min(z), np.max(z), nz)
        BX_interp = RegularGridInterpolator((X,Y,Z), Bx.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        BY_interp = RegularGridInterpolator((X,Y,Z), By.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        BZ_interp = RegularGridInterpolator((X,Y,Z), Bz.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        return lambda x: np.array([BX_interp(x),BY_interp(x),BZ_interp(x)]).T
    


# %%
class SensorCollection:
    def __init__(self, *sensors):
        self.sensors = []
        self.addSensor(*sensors)
            
    def addSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,Sensor) and s not in self.sensors:
                self.sensors.append(s)
            elif isinstance(s, SensorCollection):
                self.addSensor(*s.sensors)
        
    def removeSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,Sensor) and s in self.sensors:
                self.sensors.remove(s)
            elif isinstance(s, SensorCollection):
                self.removeSensor(*s.sensors)
                
    def getB(self, *sources):
        return np.array([s.getB(*sources) for s in self.sensors])
            
    def getPositions(self):
        pos=[]
        for s in self.sensors:
            pos.append(s.position)
        return np.array(pos)
    
    def getAngles(self):
        ang=[]
        for s in self.sensors:
            ang.append(s.angle)
        return np.array(ang)
    
    def getAxis(self):
        ax=[]
        for s in self.sensors:
            ax.append(s.axis)
        return np.array(ax)
    
    def getPosAngAx(self):
        paa=[]
        for s in self.sensors:
            paa.append([np.array(s.position), s.angle, np.array(s.axis)])
        return np.array(paa)
        
    def move(self, displacement):
        for s in self.sensors:
            s.move(displacement)
    
    @property
    def position(self):
        '''returns the barycenter of the collection items positions'''
        return np.mean([s.position for s in self.sensors],axis=0)
    
    def rotate(self, angle, axis, anchor='barycenter'):
        if anchor=='barycenter':
            anchor = self.position
        for s in self.sensors:
            s.rotate(angle, axis, anchor=anchor)
            
    def __add__(self, other):
        assert isinstance(other, (SensorCollection, Sensor)) , str(other) +  ' item must be a SensorCollection or a sensor'
        if not isinstance(other, (SensorCollection)):
            sens = [other]
        else:
            sens = other.sensors
        return SensorCollection(self.sensors + sens)

    def __sub__(self, other):
        assert isinstance(other, (SensorCollection, Sensor)) , str(other) +  ' item must be a SensorCollection or a sensor'
        if not isinstance(other, (SensorCollection)):
            sens = [other]
        else:
            sens = other.sensors
            
        col = SensorCollection(self.sensors)
        col.removeSource(sens)
        return col

        
        col = Collection(self.sources)

    def __repr__(self):
        return "\n".join([str(type(s)) for s in self.sensors])

    def __iter__(self):
        for s in self.sensors:
            yield s

    def __getitem__(self, i):
        return self.sensors[i]
