# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from magpylib._lib.mathLib_vector import angleAxisRotationV
from magpylib._lib.mathLib import angleAxisRotation
from magpylib._lib.classes.base import RCS
from magpylib._lib.classes.magnets import Box
from magpylib._lib.classes.sensor import Sensor
from magpylib._lib.classes.collection import Collection
from pathlib import Path

# %% [markdown]
# # Defaults

# %%
SENIS_file_params = dict(usecols=[1,2,3,4,5,6],
                         names=['Bx','Bz','By','x','z','y'],
                         factors=[1,-1,1,1,1,-1],
                         delimiter=',',
                         skiprows=1)
           
ANSYS_file_params = dict(usecols=[0,1,2,4,5,6],
                         names=['x','y','z','Bx','By','Bz'],
                         factors=[1000,1000,1000,1000,1000,1000],
                         delimiter=' ',
                         skiprows=2)


# %% [markdown]
# # Functions

# %%
def getBarray(*sources, POS=(0.,0.,0.), ANG=0., AXIS=(0.,0.,1.)):
    if len(sources) > 0:
        POS = np.array(POS)
        shape = POS.shape
        POS = POS.reshape(-1,3)
        if isinstance(ANG, (float,int)):
            import warnings
            warnings.warn(f'\n ANG and POS have different lengths, ({0} and {len(POS)}), repeating ANG[0] by len(POS)')
            ANG = np.tile(ANG, len(POS))
        ANG = np.array(ANG).flatten()
        AXIS = np.array(AXIS).reshape(-1,3)
        if len(AXIS)!=len(POS):
            import warnings
            warnings.warn(f'\n AXIS and POS have different lengths, ({len(AXIS)} and {len(POS)}), repeating AXIS[0] by len(POS)')
            AXIS = np.repeat([AXIS[0]], len(POS), axis=0)
        ANCHOR = POS*0  # all anchors are zeros -> rotating only Bvector, using 'anchor' to have same array shape        
        B = np.array([s.getB(POS) for s in sources]).sum(axis=0)
        Brot = angleAxisRotationV(B,-ANG, AXIS, ANCHOR)
        return Brot.reshape(shape)
    else:
        import warnings
        warnings.warn(
        "no magnetic source"
        "returning [[0,0,0]]", RuntimeWarning)
        return np.array([[0,0,0]])


# %%
def read_file(filename, recenter=None, factors=(1,1,1,1,1,1), usecols=[0,1,2,3,4,5], 
              names=['x','z','y','Bx','By','Bz'], delimiter=',', skiprows=1,
              bounds_error=False, fill_value=np.nan, **kwargs):
    def recenter_df(v):
        return v - 0.5*(v.max(axis=0) + v.min(axis=0))
    default_names = ['x','y','z','Bx','By','Bz']
    df = pd.read_csv(filename, usecols=usecols, names=default_names, delimiter=delimiter, skiprows=skiprows, **kwargs)
    if df.duplicated(['x', 'y', 'z']).any():
        df = df.groupby(['x', 'y', 'z']).median().reset_index(drop=True)
    dfc = df.copy()
    # redefine coordinate systems using factors and columns
    for dn,n,f in zip(default_names, names, factors):
        df[dn] = dfc[n].values*f
    if recenter is not None:
        for i,k in enumerate(['x','y','z']):
            df[k] = recenter_df(df[k].values) + recenter[1]
    return DiscreteSourceBox(df, bounds_error=bounds_error, fill_value=fill_value)


# %% [markdown]
# ## MCollection/MDataset

# %%
SENIS_file_params = dict(usecols=[1,2,3,4,5,6],
                         names=['Bx','Bz','By','x','z','y'],
                         factors=[1,-1,1,1,1,-1],
                         delimiter=',',
                         skiprows=1)
           
ANSYS_file_params = dict(usecols=[0,1,2,4,5,6],
                         names=['x','y','z','Bx','By','Bz'],
                         factors=[1000,1000,1000,1000,1000,1000],
                         delimiter=' ',
                         skiprows=2)


# %%
def read_file(filename, recenter=None, factors=(1,1,1,1,1,1), usecols=[0,1,2,3,4,5], 
              names=['x','z','y','Bx','By','Bz'], delimiter=',', skiprows=1,
              bounds_error=False, fill_value=np.nan, **kwargs):
    def recenter_df(v):
        return v - 0.5*(v.max(axis=0) + v.min(axis=0))
    default_names = ['x','y','z','Bx','By','Bz']
    df = pd.read_csv(filename, usecols=usecols, names=default_names, delimiter=delimiter, skiprows=skiprows, **kwargs)
    if df.duplicated(['x', 'y', 'z']).any():
        df = df.groupby(['x', 'y', 'z']).median().reset_index(drop=True)
    dfc = df.copy()
    # redefine coordinate systems using factors and columns
    for dn,n,f in zip(default_names, names, factors):
        df[dn] = dfc[n].values*f
    if recenter is not None:
        for i,k in enumerate(['x','y','z']):
            df[k] = recenter_df(df[k].values) + recenter[1]
    return DiscreteSourceBox(df, bounds_error=bounds_error, fill_value=fill_value)


# %%
class MCollection(Collection):
    def __init__(self, sources=None, sensors=None, nonmodelobjs=None, name=None):
        super().__init__(*sources)
        if name is None:
            name = 'Collection_' +str(id(self))
        self.name = name
        self.sensors = sensors if sensors is not None else []
        self.nonmodelobjs = nonmodelobjs if nonmodelobjs is not None else []

    def __repr__(self):
        return f"Magpylib Collection\n"\
               f"sources: {self.sources}\n"\
               f"sensors: {self.sensors}\n"\
               f"non model objects: {self.nonmodelobjs}\n"


# %%
class MDataset:
    def __init__(self, *collections):
        self.collections=[]
        self.add_collections(*collections)
        
    @property
    def names(self):
        return [d.name for d in self.collections]
    
    @property
    def sets(self):
        return [d.sources for d in self.collections]
    
    def add_collections(self, *collections):
        for d in collections:
            self.collections.append(d)
            
    def __iter__(self):
        for s in self.collections:
            yield s

    def __getitem__(self, i):
        return self.collections[i]
    
    def __len__(self):
        return len(self.collections)
    
    def __add__(self, other):
        assert isinstance(other, (MDataset, MCollection)) , str(other) +  ' item must be a MCollection or a MDataset'
        if not isinstance(other, (MCollection)):
            d = [other]
        else:
            d = other.collections
        return MDataset(self.collections + d)
    
    def __repr__(self):
        return f"Magpylib Dataset\n"\
               f"{str({d.name: str(len(d.sources))+' sources, ' + str(len(d.sensors))+ ' sensors, ' + str(len(d.nonmodelobjs))+ ' non model objects' for d in self.collections})}"


# %% [markdown]
# # Rotation Axis

# %%
class RotationAxis(RCS):
    def __init__(self, pos=(0.,0.,0.), angle=0., axis=(0.,0.,1.), dim=10., name=None):
        super().__init__(position=pos, angle=angle, axis=axis)
        self.dimension = dim
        if name is None:
            name = 'RotationAxis_' +str(id(self))
        self.name = name
        
    def __repr__(self):
        return f"{self.name}"\
               f"\n length: a={self.position[0]:.2f}mm, b={self.position[1]:.2f}mm, c={self.position[2]:.2f}mm"\
               f"\n position: x={self.position[0]:.2f}mm, y={self.position[1]:.2f}mm, z={self.position[2]:.2f}mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x={self.axis[0]:.2f}, y={self.axis[1]:.2f}, z={self.axis[2]:.2f}"


# %% [markdown]
# # Sensor 3d

# %%
class Sensor3d(Sensor):
    def __init__(self, pos=(0.,0.,0.), angle=0., axis=(0.,0.,1.), dim=1., name=None):
        super().__init__(pos=pos, angle=angle, axis=axis)
        self._dimension = self._set_dim(dim)
        if name is None:
            name = 'Sensor3d_' +str(id(self))
        self.name = name
    
    @property
    def dimension(self):
        return self._dimension
    
    @dimension.setter
    def dimension(self,val):
        self._dimension = self._set_dim(val)
        
    def _set_dim(self,dim):
        try:
            if len(dim)==2:
                dim = np.array([dim[0],dim[1],np.mean(dim)/5])
            else:
                dim = np.array(dim[:3])
        except:
            dim = np.array([dim, dim, dim])
        return dim
            
    def __repr__(self):
        return f"{self.name}"\
               f"\n dimension: a={self.position[0]:.2f}mm, b={self.position[1]:.2f}mm, c={self.position[2]:.2f}mm"\
               f"\n position: x={self.position[0]:.2f}mm, y={self.position[1]:.2f}mm, z={self.position[2]:.2f}mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x={self.axis[0]:.2f}, y={self.axis[1]:.2f}, z={self.axis[2]:.2f}"



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
        self.position = np.array(pos)
        self.magnetization = (0,0,0)
        self.angle = angle
        self.axis = np.array(axis)
        self.interpFunc = self._interpolate_data(df, bounds_error=bounds_error, fill_value=fill_value)
        self.data_downsampled = self.get_downsampled_array(df, N=5)
        self.dataframe = df
        
    def getB(self, pos):
        POS = np.array(pos - self.position)
        if self.angle!=0:
            if len(POS)==3:
                POS = np.array([POS])
            ANG = np.tile(self.angle, len(POS)).flatten()
            AXIS = np.repeat(np.array([self.axis]), len(POS), axis=0)
            ANCHOR = POS*0 
            POSrot = angleAxisRotationV(POS, -ANG, AXIS, ANCHOR)
            Bi = self.interpFunc(POSrot)
            B = angleAxisRotationV(Bi, ANG, AXIS, ANCHOR)
        else:
            B = self.interpFunc(POS)
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
class SensorCollection:
    def __init__(self, *sensors, pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        self.rcs = RCS(position=pos, angle=angle, axis=axis)
        self.sensors = []
        self.addSensor(*sensors)

    def __repr__(self):
        return f"SensorCollection"\
               f"\n sensor children: N={len(self.sensors)}"\
                "position: x={:.2f}mm, y={:.2f}mm, z={:.2f}mm\n".format(*self.position,) + \
                "angle: {:.2f} Degrees\n".format(self.angle) + \
                "axis: x={:.2f}, y={:.2f}, z={:.2f}".format(*self.axis)

    def __iter__(self):
        for s in self.sensors:
            yield s

    def __getitem__(self, i):
        return self.sensors[i]
    
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
        col.removeSensor(sens)
        return col
    
    
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

    @property
    def position(self):
        return self.rcs.position
    @position.setter
    def position(self, value):
        self.move(value-self.rcs.position)
        
    @property
    def angle(self):
        return self.rcs.angle
    @angle.setter
    def angle(self, value):
        self.rotate(value-self.rcs.angle, axis=self.rcs.axis, anchor=self.rcs.position)
        
    @property
    def axis(self):
        return self.rcs.axis
    @axis.setter
    def axis(self, value):
        angle = self.rcs.angle
        self.rotate(-angle, axis=self.rcs.axis, anchor=self.rcs.position)
        self.rotate(angle, axis=value, anchor=self.rcs.position)
    
    def _get_positions(self, recursive=True):
        if recursive:
            return np.array([s._get_positions(recursive=True) if isinstance(s,(SensorCollection, SurfaceSensor)) else s.position for s in self.sensors])
        else:
            return np.array([s.position for s in self.sensors])
    
    def _get_angles(self, recursive=True):
        if recursive:
            return np.array([s._get_angles(recursive=True) if isinstance(s,(SensorCollection, SurfaceSensor)) else s.angle for s in self.sensors])
        else:
            return np.array([s.angle for s in self.sensors])
    
    def _get_axes(self, recursive=True):
        if recursive:
            return np.array([s._get_axes(recursive=True) if isinstance(s,(SensorCollection, SurfaceSensor)) else s.axis for s in self.sensors])
        else:
            return np.array([s.axis for s in self.sensors])
    
    @property    
    def positions(self):
        return self._get_positions(recursive=False)
    
    @property    
    def angles(self):
        return self._get_angles(recursive=False)
    
    @property
    def axes(self):
        return self._get_axes(recursive=False)
    
    def move(self, displacement):
        self.rcs.move(displacement)
        for s in self.sensors:
            s.move(displacement)
            
    def rotate(self, angle, axis, anchor='self.position'):
        self.rcs.rotate(angle=angle, axis=axis, anchor=anchor)
        if str(anchor) == 'self.position':
            anchor = self.rcs.position
        for s in self.sensors:
            s.rotate(angle, axis, anchor=anchor)
            
    def getBarray(self, *sources):
        POS, ANG, AXIS = self._get_positions(recursive=True), self._get_angles(recursive=True), self._get_axes(recursive=True)
        return getBarray(*sources, POS=POS, ANG=ANG, AXIS=AXIS)


# %% [markdown]
# # Surface Sensor

# %%
class SurfaceSensorOld(SensorCollection):
    def __init__(self, Nelem=(3,3), dim=(0.2,0.2), pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        try:
            sensors=[Sensor(pos=(i,0,0)) for i in range(Nelem[0]*Nelem[1])]
        except:
            sensors=[Sensor(pos=(0,0,0))]
        super().__init__(*sensors, pos=pos, angle=angle, axis=axis)
        self.update(Nelem=Nelem, dim=dim)

    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, val):
        self.update(dim=val)
        
    @property
    def Nelem(self):
        return self._Nelem
    @Nelem.setter
    def Nelem(self, val):
        self.update(Nelem=val)
        
    def update(self, pos=None, angle=None, axis=None, dim=None, Nelem=None):
        if pos is not  None:
            self.rcs.position = pos
        if angle is not  None:
            self.rcs.angle = angle
        if axis is not  None:
            self.rcs.axis = axis
        if dim is None:
            dim = self._dimension
        if isinstance(dim, (int,float)):
            dim = (dim, dim)
        dim = self._dimension = np.array(dim)
        if Nelem is None:
            Nelem = self._Nelem
        if isinstance(Nelem, (int,float)):
            n1 = np.int(np.sqrt(Nelem))
            n2 = np.int(Nelem/n1)
            Nelem = (n1, n2)
        self._Nelem = Nelem = np.array(Nelem).astype(int)
        
        if Nelem[0]==1 or Nelem[1]==1:
            dim = self._dimension = np.array([0,0])
        
        POS = np.mgrid[-dim[0]/2:dim[0]/2:Nelem[0]*1j,-dim[1]/2:dim[1]/2:Nelem[1]*1j, 0:0:1j].reshape(3,-1).T
        ANG = np.tile(self.angle, len(POS))
        AXIS = np.repeat([self.axis], len(POS), axis=0)
        ANCHOR = np.repeat([[0,0,0]], len(POS), axis=0)
        posrot = angleAxisRotationV(POS=POS ,ANG=ANG, AXIS=AXIS, ANCHOR=ANCHOR)
        
        for i in range(len(posrot)):
            if i>=len(self.sensors):
                self.addSensor(Sensor(pos=(i,0,0)))
            self.sensors[i].position = posrot[i] + self.position
            self.sensors[i].angle= self.angle
            self.sensors[i].axis = self.axis
            i+=1
        if i<len(self.sensors):
            self.removeSensor(*self.sensors[i:])
    
    def getB(self, *sources):
        return self.getBarray(*sources).mean(axis=0)
    
    def __repr__(self):
        return f"name: SurfaceSensor"\
               f"\n surface elements: Nx={self.Nelem[0]}, Ny={self.Nelem[1]}"\
               f"\n dimension x: {self.dimension[0]:.2f}, mm, y: {self.dimension[1]:.2f}, mm"\
               f"\n position x: {self.position[0]:.2f}, mm, y: {self.position[1]:.2f}, mm z: {self.position[2]:.2f} mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x: {self.axis[0]:.2f}, y: {self.axis[1]:.2f}, z: {self.axis[2]:.2f}"


# %%
class SurfaceSensor(RCS):
    def __init__(self, Nelem=(3,3), dim=(0.2,0.2), pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        super().__init__(position=pos, angle=angle, axis=axis)
        self.update(Nelem=Nelem, dim=dim)

    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, val):
        self.update(dim=val)
        
    @property
    def Nelem(self):
        return self._Nelem
    @Nelem.setter
    def Nelem(self, val):
        self.update(Nelem=val)
    
    @property    
    def positions(self):
        return self._get_positions()
    
    def _get_positions(self, recursive=True):
        self.update()
        return self._positions
    
    def _get_angles(self, recursive=True):
        self.update()
        return self._angles
    
    def _get_axes(self, recursive=True):
        self.update()
        return self._axes
        
    def update(self, pos=None, angle=None, axis=None, dim=None, Nelem=None):
        if pos is not  None:
            self.position = pos
        if angle is not  None:
            self.angle = angle
        if axis is not  None:
            self.axis = axis
        if dim is None:
            dim = self._dimension
        if isinstance(dim, (int,float)):
            dim = (dim, dim)
        dim = self._dimension = np.array(dim)
        if Nelem is None:
            Nelem = self._Nelem
        if isinstance(Nelem, (int,float)):
            n1 = np.int(np.sqrt(Nelem))
            n2 = np.int(Nelem/n1)
            Nelem = (n1, n2)
        self._Nelem = Nelem = np.array(Nelem).astype(int)
        
        if Nelem[0]==1 or Nelem[1]==1:
            dim = self._dimension = np.array([0,0])
        
        POS = np.mgrid[-dim[0]/2:dim[0]/2:Nelem[0]*1j,-dim[1]/2:dim[1]/2:Nelem[1]*1j, 0:0:1j].reshape(3,-1).T
        self._angles = ANG = np.tile(self.angle, len(POS))
        self._axes = AXIS = np.repeat([self.axis], len(POS), axis=0)
        ANCHOR = np.repeat([[0,0,0]], len(POS), axis=0)
        posrot = angleAxisRotationV(POS=POS ,ANG=ANG, AXIS=AXIS, ANCHOR=ANCHOR)
        self._positions = posrot + self.position
    
    def getB(self, *sources):
        return self.getBarray(*sources).mean(axis=0)
    
    def getBarray(self, *sources):
        return getBarray(*sources, POS=self.positions, ANG=self.angle, AXIS=self.axis)
        
    def __repr__(self):
        return f"name: SurfaceSensor"\
               f"\n surface elements: Nx={self.Nelem[0]}, Ny={self.Nelem[1]}"\
               f"\n dimension x: {self.dimension[0]:.2f}, mm, y: {self.dimension[1]:.2f}, mm"\
               f"\n position x: {self.position[0]:.2f}, mm, y: {self.position[1]:.2f}, mm z: {self.position[2]:.2f} mm"\
               f"\n angle: {self.angle:.2f} Degrees"\
               f"\n axis: x: {self.axis[0]:.2f}, y: {self.axis[1]:.2f}, z: {self.axis[2]:.2f}"


# %% [markdown]
# # Circular Sensor Array

# %%
class CircularSensorArray(SensorCollection):
    def __init__(self, Rs=1, elem_dim=(0.2,0.2), Nelem=(3,3), num_of_sensors=4, start_angle=0):
        self.start_angle = start_angle
        self.elem_dim = elem_dim
        self.Nelem = Nelem
        self.Rs = Rs
        S = [SurfaceSensor(pos=(i,0,0), dim=elem_dim, Nelem=Nelem) for i in range(num_of_sensors)]
        super().__init__(*S)
        self.initialize(Rs=Rs, start_angle=start_angle, elem_dim=elem_dim)
    
    def initialize(self, Rs, start_angle=None, elem_dim=None, Nelem=None):
        if Rs is None:
            Rs = self.Rs
        else:
            self.Rs = Rs
        if start_angle is None:
            start_angle= self.start_angle
        else:
            self.start_angle = start_angle
        if elem_dim is None:
            elem_dim= self.elem_dim
        else:
            self.elem_dim = elem_dim
        if Nelem is None:
            Nelem= self.Nelem
        else:
            self.Nelem = Nelem
                
        theta = np.deg2rad(np.linspace(start_angle, start_angle+360, len(self.sensors)+1))[:-1]
        for s,t in zip(self.sensors,theta):
            s.update(pos = (Rs*np.cos(t), Rs*np.sin(t),0),
                     angle = 0,
                     axis = (0,0,1), 
                     dim = elem_dim,
                     Nelem = Nelem
                    )


# %% [markdown]
# # Testing

# %% [raw]
# ss = SurfaceSensor(Nelem=(30,30), dim=(8,8), pos=(0,0,15), angle=90, axis=(1,0,0))

# %% [raw]
# ds = DiscreteSourceBox('data/discrete_source_data.csv')
# box = Box(dim=(10,10,10), mag=(1,0,0), pos=(10,0,0))
# s = Sensor()
# ss = SurfaceSensor(Nelem=(10,10), dim=0.03)
# print(s.getB(ds), ss.getB(ds))
#
# csa = CircularSensorArray(Rs=2, num_of_sensors=1, Nelem=(5,5), start_angle=180, elem_dim=(0.2,0.2))
#
# %time csa.getBarray(box).mean(axis=1)

# %% [raw]
# def f1(N=1):
#     csa = CircularSensorArray(Rs=2, num_of_sensors=4, Nelem=(5,5), start_angle=180, elem_dim=(0.2,0.2))
#     B = []
#     for i in range(N):
#         csa.rotate(angle=i, axis=(1,2,3))
#         ANG = csa._get_angles()
#         POS = csa._get_positions()
#         AXIS = csa._get_axes()
#         B.append(getBarray(box, POS=POS, ANG=ANG, AXIS=AXIS))
#     #return np.array(B).mean(axis=2)
#
#
# def f2(N=1):
#     csa = CircularSensorArray(Rs=2, num_of_sensors=4, Nelem=(5,5), start_angle=180, elem_dim=(0.2,0.2))
#     ANG = []
#     AXIS = []
#     POS = []
#     for i in range(N):
#         csa.rotate(angle=i, axis=(1,2,3))
#         POS.append(csa._get_positions())
#     B = getBarray(box, POS=POS, ANG=csa.angle, AXIS=csa.axis)
#     
#     #return np.array(B.mean(axis=2))
#
#
# def f3(N=1):
#     csa = CircularSensorArray(Rs=2, num_of_sensors=4, Nelem=(5,5), start_angle=180, elem_dim=(0.2,0.2))
#     B = []
#     for i in range(N):
#         csa.rotate(angle=i, axis=(1,2,3))
#         B.append(np.array([s.getB(box) for s in csa.sensors]))
#     #return np.array(B)
#
#
#     
#
# N=100
# %time f1(N)
# %time f2(N)
# %time f3(N)
