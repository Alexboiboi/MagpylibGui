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
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from magpylib._lib.mathLib import angleAxisRotation
from magpylib._lib.classes.base import RCS
from magpylib._lib.classes.magnets import Box
from magpylib._lib.classes.sensor import Sensor
from pathlib import Path


# %% [markdown]
# # Discrete Source Box

# %%
class DiscreteSourceBox(Box):
    def __init__(self, data, bounds_error=None, fill_value=None, pos=(0.,0.,0.), angle=0., axis=(0.,0.,1.)):
        '''
        data : csv file, pandas dataframe, or numpy array
            !!! IMPORTANT !!! columns value must be in this order: x,y,z,Bx,By,Bz
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used.
        fill_value : number, optional
            If provided, the value to use for points outside of the interpolation domain. If None, values outside the domain are extrapolated.
        '''
        
        try:
            Path(data).is_file()
            df = pd.read_csv(data)
        except:
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data, columns=['x','y','z','Bx','By','Bz'])
                
        df = df.sort_values(['x','y','z'])    
        m = np.min(df.values,axis=0)
        M = np.max(df.values,axis=0)
        self.dimension = (M-m)[:3]
        self._center = 0.5*(M+m)[:3]
        self.position = self._center + np.array(pos)
        self.position = self._center + np.array(pos)
        self.magnetization = (0,0,0)
        self.angle = angle
        self.axis = axis
        self.interpFunc = self._interpolate_data(df, bounds_error=bounds_error, fill_value=fill_value)
        self.data_downsampled = self.get_downsampled_array(df, N=5)
        self.dataframe = df
        
    def getB(self, pos):
        pos += self._center - self.position
        B = self.interpFunc(pos)
        if self.angle!=0:
            return np.array([angleAxisRotation(b, self.angle, self.axis) for b in B])
        else:
            return B[0] if B.shape[0] == 1 else B
    
    def _interpolate_data(self, data, bounds_error, fill_value):
        '''data: pandas dataframe
            x,y,z,Bx,By,Bz dataframe sorted by (x,y,z)
        returns: 
            interpolating function for B field values'''
        x,y,z,Bx,By,Bz = data.values.T
        nx,ny,nz = len(np.unique(x)), len(np.unique(y)), len(np.unique(z))
        X = np.linspace(np.min(x), np.max(x), nx)
        Y = np.linspace(np.min(y), np.max(y), ny)
        Z = np.linspace(np.min(z), np.max(z), nz)
        BX_interp = RegularGridInterpolator((X,Y,Z), Bx.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        BY_interp = RegularGridInterpolator((X,Y,Z), By.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        BZ_interp = RegularGridInterpolator((X,Y,Z), Bz.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
        return lambda x: np.array([BX_interp(x),BY_interp(x),BZ_interp(x)]).T
    
    def get_downsampled_array(self, df, N=5):
        '''
        df : pandas dataframe 
            x,y,z,Bx,By,Bz dataframe sorted by (x,y,z)
        N : integer
            number of points per dimensions left after downsampling, 
            min=2 if max>len(dim) max=len(dim)
        returns:
            downsampled numpy array'''
        df=df.copy()
        l = df.shape[0]
        df['Bmag'] =( df['Bx']**2 + df['By']**2 + df['Bz']**2)**0.5
        masks=[]
        N=1 if N<1 else N
        for i,k in enumerate(['x','y', 'z']):
            u = df[k].unique()
            dsf = int(len(u)/N) 
            if dsf<1:
                dsf = 1
            masks.append(df[k].isin(u[::dsf]))
        dfm = df[masks[0]&masks[1]&masks[2]]
        data = dfm[['x','y','z','Bmag']].values
        return data
    
    def __repr__(self):
        return "DiscreteSourceBox\n" + \
                "dimensions: a={:.2f}mm, b={:.2f}mm, c={:.2f}mm\n".format(*self.dimension) + \
                "position: x={:.2f}mm, y={:.2f}mm, z={:.2f}mm\n".format(*self.position,) + \
                "angle: {:.2f} Degrees\n".format(self.angle) + \
                "axis: x={:.2f}, y={:.2f}, z={:.2f}".format(*self.axis)



# %% [markdown]
# # Sensor Collection

# %%
class SensorCollection(RCS):
    def __init__(self, *sensors, pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        super().__init__(position=pos, angle=angle, axis=axis)
        self.sensors = []
        self.addSensor(*sensors)

    def __repr__(self):
        return f"SensorCollection"\
               f"\n sensor children: N={len(self.sensors)}"\
               f"\n position x: {self.position[0]:.2f} mm  n y: {self.position[1]:.2f}mm z: {self.position[2]:.2f}mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x: {self.axis[0]:.2f}   n y: {self.axis[1]} z: {self.axis[2]}"

    def __iter__(self):
        for s in self.sensors:
            yield s

    def __getitem__(self, i):
        return self.sensors[i]
            
    def addSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,(Sensor,SurfaceSensor)) and s not in self.sensors:
                self.sensors.append(s)
            elif isinstance(s, SensorCollection):
                self.addSensor(*s.sensors)
        
    def removeSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,(Sensor,SurfaceSensor)) and s in self.sensors:
                self.sensors.remove(s)
            elif isinstance(s, SensorCollection):
                self.removeSensor(*s.sensors)
                
    def getB(self, *sources, mean=False):
        B = np.array([s.getB(*sources) for s in self.sensors])
        if mean:
            return B.mean(axis=0)
        else:
            return B
            
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
        super().move(displacement)
        for s in self.sensors:
            s.move(displacement)
            
    def rotate(self, angle, axis, anchor='self.position'):
        super().rotate(angle=angle, axis=axis, anchor=anchor)
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


# %%
class SensorCollection(RCS):
    def __init__(self, *sensors, pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        super().__init__(position=pos, angle=angle, axis=axis)
        self.sensors = []
        self.addSensor(*sensors)

    def __repr__(self):
        return f"SensorCollection"\
               f"\n sensor children: N={len(self.sensors)}"\
               f"\n position x: {self.position[0]:.2f} mm  n y: {self.position[1]:.2f}mm z: {self.position[2]:.2f}mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x: {self.axis[0]:.2f}   n y: {self.axis[1]} z: {self.axis[2]}"

    def __iter__(self):
        for s in self.sensors:
            yield s

    def __getitem__(self, i):
        return self.sensors[i]
            
    def addSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,(Sensor,SurfaceSensor)) and s not in self.sensors:
                self.sensors.append(s)
            elif isinstance(s, SensorCollection):
                self.addSensor(*s.sensors)
        
    def removeSensor(self, *sensors):
        for s in sensors:
            if isinstance(s,(Sensor,SurfaceSensor)) and s in self.sensors:
                self.sensors.remove(s)
            elif isinstance(s, SensorCollection):
                self.removeSensor(*s.sensors)
                
    def getB(self, *sources, mean=False):
        B = np.array([s.getB(*sources) for s in self.sensors])
        if mean:
            return B.mean(axis=0)
        else:
            return B
        
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
        super().move(displacement)
        for s in self.sensors:
            s.move(displacement)
            
    def rotate(self, angle, axis, anchor='self.position'):
        super().rotate(angle=angle, axis=axis, anchor=anchor)
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


# %% [markdown]
# # Surface Sensor

# %%
class SurfaceSensor(SensorCollection):
    def __init__(self, N=(20,30), dim=(0.2,0.3), pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        self.dimension = dim
        self.Nx = N[0]
        self.Ny = N[1]
        self.pos_x = np.linspace(-dim[0]/2, dim[0]/2, N[0])
        self.pos_y = np.linspace(-dim[1]/2, dim[1]/2, N[1])
        sensors=[Sensor(pos=(i,0,0)) for i in range(N[0]*N[1])]
        super().__init__(*sensors, pos=pos, angle=angle, axis=axis)
        self._update()
    
    def _update(self, dim=None, pos=None, angle=None, axis=None):
        self.position = pos if pos is not None else self.position
        self.dimension = dim if dim is not None else self.dimension
        self.angle = angle if angle is not None else self.angle
        self.axis = axis if axis is not None else self.axis
        i=0
        for nx in self.pos_x:
            for ny in self.pos_y:
                self.sensors[i].setPosition((nx,ny,0))
                self.sensors[i].setOrientation(angle=0, axis=(0,0,1))
                self.sensors[i].rotate(angle=self.angle, axis=self.axis, anchor=(0,0,0))
                self.sensors[i].move(self.position)
                i+=1
    
    def getB(self, *sources, mean=True):
        return super().getB(*sources, mean=mean)
    
    def __repr__(self):
        return f"name: SurfaceSensor"\
               f"\n surface elements: Nx={self.Nx}, Ny={self.Ny}"\
               f"\n dimension x: {self.dimension[0]:.2f} mm  n y: {self.dimension[1]:.2f}mm"\
               f"\n position x: {self.position[0]:.2f} mm  n y: {self.position[1]:.2f}mm z: {self.position[2]:.2f}mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x: {self.axis[0]:.2f}   n y: {self.axis[1]:.2f} z: {self.axis[2]:.2f}"


# %% [raw]
# ss = SurfaceSensor(N=(3,3), dim=(8,8), pos=(0,0,15), angle=90, axis=(1,0,0))
# [s.position for s in ss.sensors]

# %% [markdown]
# # Circular Sensor Array

# %%
class CircularSensorArray(SensorCollection):
    def __init__(self, Rs=1, elem_dim=(0.2,0.2), Nelem=(20,30), num_of_sensors=4, start_angle=0, surface_sensor=False):
        self.start_angle = start_angle
        self.elem_dim = elem_dim
        if surface_sensor:
            S = [SurfaceSensor(pos=(i,0,0), dim=elem_dim, N=Nelem) for i in range(num_of_sensors)]
        else:
            S = [Sensor(pos=(i,0,0)) for i in range(num_of_sensors)]
        super().__init__(*S)
        self.setSensorsDim(elem_dim)
        self.initialize(Rs)
    
    def initialize(self, Rs, start_angle='default', elem_dim='default'):
        if start_angle == 'default':
            start_angle= self.start_angle
        if elem_dim == 'default':
            elem_dim= self.elem_dim
        theta = np.deg2rad(np.linspace(start_angle, start_angle+360, len(self.sensors)+1))[:-1]
        for s,t in zip(self.sensors,theta):
            pos = (Rs*np.cos(t), Rs*np.sin(t),0)
            s._update(pos=pos, angle=0, axis=(0,0,1), dim=elem_dim)
            
    
    def setSensorsDim(self, elem_dim):
        assert all(i >= 0 for i in elem_dim)>0, 'dim must be positive'
        for s in self.sensors:
            s.dimension = elem_dim

# %% [markdown]
# # Testing

# %% [raw]
# ds = DiscreteSourceBox('data/discrete_source_data.csv')
# s = Sensor()
# ss = SurfaceSensor()
# #s.getB(ds), ss.getB(ds)

# %%
