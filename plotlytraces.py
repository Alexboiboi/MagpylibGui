# -*- coding: utf-8 -*-
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

# %% [markdown]
# # Imports

# %%
import numpy as np
import pandas as pd
import magpylib as magpy
import plotly.graph_objects as go
from plotly.figure_factory import create_streamline

from magpylib._lib.mathLib import angleAxisRotation
from magpylib._lib.classes.magnets import Box, Cylinder, Sphere
from magpylib._lib.classes.currents import Line, Circular
from magpylib._lib.classes.moments import Dipole
from magpylib._lib.classes.sensor import Sensor
from magpylib._lib.classes.collection import Collection

import magpylibutils
from magpylibutils import (DiscreteSourceBox, SensorCollection, SurfaceSensor, MCollection,
                           RotationAxis, 
                           Streamlines, Surface, 
                           isSource)

# Defaults
SENSORSIZE = 1
DIPOLESIZEREF = 5
DISCRETESOURCE_OPACITY = 0.1


# %% [markdown]
# # Special functions

# %%
def _getIntensity(points, mag, pos):
    '''points: [x,y,z] array'''
    if sum(mag)!=0:
        p = np.array(points)
        pos = np.array(pos)
        m = np.array(mag) /   np.linalg.norm(mag)
        a = ((p[0]-pos[0])*m[0] + (p[1]-pos[1])*m[1] + (p[2]-pos[2])*m[2])
        b = (p[0]-pos[0])**2 + (p[1]-pos[1])**2 + (p[2]-pos[2])**2
        return a /   np.sqrt(b)
    else:
        return points*0

def _getColorscale(cst=0.1):
    return [[0, 'turquoise'], [0.5*(1-cst), 'turquoise'],[0.5*(1+cst), 'magenta'], [1, 'magenta']]

def repnan(a):
    '''replaces nan values from a numpy array by None'''
    a = np.array(a)
    a[np.isnan(a)]=None
    return a


# %% [markdown]
# # Other definitions

# %%
def make_RotationAxis(rotAxis, name='Rotation Axis', **kwargs):
    pos = rotAxis.position
    dim = rotAxis.dimension
    angle = rotAxis.angle
    axis = rotAxis.axis
    try:
        kwargs.pop('color')
    except:
        pass
    x = [pos[0], pos[0]]
    y = [pos[1], pos[1]]
    z = [pos[2]-dim/2, pos[2]+dim/2]
    points = np.array([x,y,z])
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T

    scatter = go.Scatter3d(x=x, y=y, z=z, name=name, mode='lines+text', text=['(-)','(+)'], textposition="top right")
    scatter.update(**kwargs)
    return scatter  


# %% [markdown]
# # Sources definitions
#

# %% [markdown]
# ## Surface/Streamlines

# %%
def make_Surface(surfsens, sensorsources=None, sensoraxis='z', color=None, **kwargs):
    '''
    sensorsources: iterable of magpylib sources
        in order to make the surface plot the Bfield values need to be retrieved for the corresponding sources
    sensoraxis: string
        one of "x", "y", "z" corresponding to the relative coordinate system of the sensor array
    '''
    colorscale=[[0, color], [1,color]] if color is not None else None
    x,y,z = surfsens.positions.T.reshape(3, *surfsens.Nelem)
    if sensorsources is not None:
        B = surfsens.getBarray(*sensorsources).reshape(*surfsens.Nelem, 3)
        surfacecolor = B[:,:,['x','y','z'].index(sensoraxis)]
        colorscale = None
    else:
        surfacecolor = x*0
    trace = go.Surface(x=x,y=y,z=z, surfacecolor=surfacecolor, colorscale=colorscale,
                       name=f'surface ({sensoraxis}-axis, {len(sensorsources)} sources)')
    trace.update(**kwargs)
    return trace

def make_Streamlines(surfsens, density='auto', arrow_scale=0.1, sensorsources=None, color=None, **kwargs):
    '''
    sensorsources: iterable of magpylib sources
        in order to make the streamlines plot the Bfield values need to be retrieved for the corresponding sources
    '''
    if sensorsources is None:
        sensorsources=[]
        
    angle = surfsens.angle
    pos = surfsens.position
    axis = surfsens.axis
    
    density = surfsens.Nelem[0]/4 if density=='auto' else density

    B = surfsens.getBarray(*sensorsources).reshape(*surfsens.Nelem, 3)
    surfsens.position = np.array([0,0,0])
    surfsens.angle = 0
    x,y,z = surfsens.positions.T.reshape(3, *surfsens.Nelem)
    xs, ys = x[:,0], y.T[:,0]
    surfsens.angle = angle
    surfsens.position = pos
    surfsens.axis = axis

    Bx, By = B[:,:,0], B[:,:,1]
    streamline = create_streamline(x=xs, y=ys , u=Bx.T, v=By.T, arrow_scale=arrow_scale, density=density)
    sl = streamline.data[0]
    points = np.array([sl.x, sl.y, np.zeros(len(sl.x))])
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=[0,0,0]) for p in points.T]).T
    else:
        x,y,z = points
    trace = go.Scatter3d(x=repnan(x)+pos[0], y=repnan(y)+pos[1], z=repnan(z)+pos[2], 
                         mode = 'lines',
                         name=f'streamlines ({len(sensorsources)} sources)')
    trace.update(**kwargs)
    return trace


# %% [markdown]
# ## Sensor

# %%
def makeSensor(pos = (0,0,0), angle=0, axis=(0,0,1), dim=1, showlegend=True, **kwargs):
    box = go.Mesh3d(
        i = np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j = np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k = np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        showscale=False, showlegend=showlegend,
        name='3d sensor'
    )
    try:
        if len(dim)==2:
            dim = np.array([dim[0],dim[1],np.mean(dim)/5])
        else:
            dim = np.array(dim)
    except:
        dim = np.array([dim, dim, dim])
        
    dd = 0.8 # shape modifier 
    x = np.array([-1, -1, 1, 1, -dd*1, -dd*1, dd*1, dd*1])*0.5*dim[0]+pos[0]
    y = np.array([-1, 1, 1, -1, -dd*1, dd*1, dd*1, -dd*1])*0.5*dim[1]+pos[1]
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])*0.5*dim[2]+pos[2]
    points = np.array([x,y,z])
    
    if angle!=0:
        points = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
    
    box.x , box.y, box.z = points
    box.update(**kwargs)
    return box


# %% [markdown]
# ## Discrete Box

# %%
def makeDiscreteBox(data, pos = (0,0,0), angle=0, axis=(0,0,1), showlegend=True, **kwargs):
    dmin = np.min(data,axis=0)
    dmax = np.max(data,axis=0)
    dim = (dmax - dmin)[:-1]
    box = go.Mesh3d(
        i = np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j = np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k = np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        showscale=False, showlegend=showlegend,
        name=f'''discrete data (Bmin={dmin[-1]:.2f}, Bmax={dmax[-1]:.2f}mT)'''    
    )
    x = np.array([-1, -1, 1, 1, -1, -1, 1, 1])*0.5*dim[0]+pos[0] + 0.5*(dmin[0]+dmax[0])
    y = np.array([-1, 1, 1, -1, -1, 1, 1, -1])*0.5*dim[1]+pos[1] + 0.5*(dmin[1]+dmax[1])
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])*0.5*dim[2]+pos[2] + 0.5*(dmin[2]+dmax[2])
    points = np.array([x,y,z])
    
    if angle!=0:
        points = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
    
    box.x , box.y, box.z = points
    box.update(**kwargs)
    return box


# %% [markdown]
# ## Box

# %%
def makeBox(mag=(0,0,1),  dim = (10,10,10), pos = (0,0,0), angle=0, axis=(0,0,1), cst=0.1, showlegend=True, **kwargs):
    box = go.Mesh3d(
        i = np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j = np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k = np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        showscale=False, showlegend=showlegend,
        name=f'''box ({dim[0]:.1f}x{dim[1]:.1f}x{dim[2]:.1f}mm)'''
    )
    x = np.array([-1, -1, 1, 1, -1, -1, 1, 1])*0.5*dim[0]+pos[0]
    y = np.array([-1, 1, 1, -1, -1, 1, 1, -1])*0.5*dim[1]+pos[1]
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])*0.5*dim[2]+pos[2]
    points = np.array([x,y,z])
    
    if cst is not False:
        box.colorscale = _getColorscale(cst)
        box.intensity = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    if angle!=0:
        points = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
    
    box.x , box.y, box.z = points
    box.update(**kwargs)
    return box


# %% [markdown]
# ## Cylinder

# %%
def makeCylinder(mag=(0,0,1), dim = (10,10,0), pos = (0,0,0), angle=0, axis=(0,0,1), cst=0.1, color=None, N=40, showlegend=True, **kwargs):
    dim=np.array(dim)
    if len(dim)==2:
        dim = np.array(list(dim[0:2]) + [0])
    elif len(dim) == 3 and dim[2]==0:
        dim[2] = 1e-5
    ri = min(dim[0]/2,dim[2]/2)
    ro = max(dim[0]/2,dim[2]/2)
    hmin, hmax = -dim[1]/2, dim[1]/2
     
    h = [hmin,hmin,hmax,hmax,hmin]
    s =   np.linspace(0, 2*np.pi, N)
    sa, ha =   np.meshgrid(s, h)

    ro = dim[0]/2  ; ri = dim[2]/2
    x = ro * np.cos(sa)
    y = ro * np.sin(sa)
    z = ha

    x[0] = x[-2] = x[-1] = ri * np.cos(s)
    y[0] = y[-2] = y[-1] = ri * np.sin(s)
    x,y,z = x+pos[0], y+pos[1], z+pos[2]
    def_name=f'''cylinder (d={dim[0]:.1f}, h={dim[1]:.1f}mm)'''
    cylinder=go.Surface(x=x, y=y, z=z, showscale=False, name=def_name, showlegend=showlegend)
    if cst is not False:
        cylinder.colorscale = _getColorscale(cst)
        cylinder.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    elif color is not None:
        cylinder.colorscale = [[0,color],[1,color]]
    if angle!=0:
        points = np.array([x.flatten(),y.flatten(),z.flatten()])
        xr,yr,zr = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        cylinder.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    cylinder.update(**kwargs)
    return cylinder


# %% [markdown]
# ## Sphere

# %%
def makeSphere(mag=(0,0,1), dim = 10, pos = (0,0,0), angle=0, axis=(0,0,1), cst=0.1, color=None, N=40, showlegend=True, **kwargs):
    r = min(dim/2,dim/2)
    s =   np.linspace(0, 2*np.pi, 2*N)
    t =   np.linspace(0, np.pi, N)
    tGrid, sGrid =   np.meshgrid(s, t)

    x = r * np.cos(sGrid) * np.sin(tGrid)  
    y = r * np.sin(sGrid) * np.sin(tGrid)  
    z = r * np.cos(tGrid)                

    x,y,z = x+pos[0], y+pos[1], z+pos[2]
    def_name=f'''sphere (d={dim:.1f}mm)'''
    sphere=go.Surface(x=x, y=y, z=z, showscale=False, name=def_name, showlegend=showlegend)
    
    if cst is not False:
        sphere.colorscale = _getColorscale(cst)
        sphere.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    elif color is not None:
        sphere.colorscale = [[0,color],[1,color]]
    if angle!=0:
        points = np.array([x.flatten(),y.flatten(),z.flatten()])
        xr,yr,zr = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        sphere.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    sphere.update(**kwargs)
    return sphere


# %% [markdown]
# ## Dipole

# %%
def makeDipole(moment=(0.0, 0.0, 1), pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), color=None, sizeref=2, showlegend=True, **kwargs):
    x,y,z = np.array([[p] for p in pos])
    mom_mag = np.linalg.norm(moment)
    mom = np.array(moment)/mom_mag
    if color is None:
        color = 'rgb(33,33,33)'
    
    if angle!=0:
        mom = angleAxisRotation(mom, angle, axis, anchor=pos)
     
    u,v,w = [[m] for m in moment]
    dipole = go.Cone(x=x,y=y,z=z, u=u,v=v,w=w,
                     colorscale=[[0, color], [1,color]],
                     sizeref = sizeref,  sizemode = 'absolute', showscale=False, showlegend=showlegend, 
                     name=f'''dipole ({mom_mag:.2f}T/m³)''')    
    dipole.update(**kwargs)
    return dipole


# %% [markdown]
# ## Line Current

# %%
def makeLine(curr=0.0, vertices=[(-1.0, 0.0, 0.0),(1.0,0.0,0.0)], pos=(0.0, 0.0, 0.0), N=12, angle=0.0, axis=(0.0, 0.0, 1.0), sizeref=None, color=None, showlegend=True, **kwargs):
    vert = np.array(vertices)
    Nv = len(vertices)
    points = []
    moments = []
    for i in range(Nv-1):
        pts = np.linspace(vert[i], vert[i+1], N)
        moms = np.tile((vert[i+1]-vert[i])*np.sign(curr),(N,1))
        points += [*pts]
        moments += [*moms]
    
    x,y,z = points = (np.array(points) + np.array(pos)).T
    u,v,w = moments = np.array(moments).T
        
    if color is None:
        color = 'rgb(33,33,33)'
    if sizeref is None:
        sizeref = np.sqrt(np.power(np.diff(vert, axis=0),2).sum(axis=1)).sum()
        
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        u,v,w = np.array([angleAxisRotation(p, angle, axis, anchor=(0,0,0)) for p in moments.T]).T
        
    lineCurrent = go.Cone(x=x,y=y,z=z,u=u,v=v,w=w, 
                          sizeref = sizeref, sizemode = 'absolute', showscale =False, showlegend=showlegend,
                          colorscale=[[0, color], [1,color]],
                          name=f'line current ({curr:.2f}A)')    
    lineCurrent.update(**kwargs)
    return lineCurrent


# %% [markdown]
# ## Circular Current

# %%
def makeCircular(curr=0.0, dim=1.0, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), color=None, N=40, sizeref=1, showlegend=True, **kwargs):
    t =   np.linspace(0, 2*  np.pi, N)
    x = dim/2 * np.cos(t) + pos[0]
    y = dim/2 * np.sin(t) + pos[1]
    z =   np.ones(N)*pos[2]
    points = np.array([x,y,z])
    u = -np.ones(N)*np.sin(t)*np.sign(curr)
    v = np.ones(N)*np.cos(t)*np.sign(curr)
    w = np.ones(N)*0
    if color is None:
        color = 'rgb(33,33,33)'
    
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        moments = np.array([u,v,w])
        u,v,w = np.array([angleAxisRotation(p, angle, axis, anchor=(0,0,0)) for p in moments.T]).T
        
    circularCurrent = go.Cone(x=x,y=y,z=z, u=u,v=v,w=w, 
                              sizeref = sizeref, sizemode = 'absolute', showscale =False, showlegend=showlegend,
                              colorscale=[[0, color], [1,color]],
                              name=f'circular current ({curr:.2f}A)')    
    circularCurrent.update(**kwargs)
    return circularCurrent


# %% [markdown]
# # Get Trace function

# %%
def getTraces(*input_objs, sensorsources='from_input', cst=0, color=None, Nver=40, 
              showhoverdata=True, dipolesizeref=DIPOLESIZEREF, 
              opacity='default', showlegend=True, 
              sensorsize=SENSORSIZE, sensoraxis='z', streamlines=False, **kwargs):
    traces=[]
    for s in input_objs:
        if isinstance(s, (tuple, list, Collection, SensorCollection, MCollection)):
            if isinstance(s, Collection):
                parent = s.sources
            elif if isinstance(s, MCollection):
                parent = s.objects
            tcs = getTraces(*parent, sensorsources=sensorsources, cst=cst, color=color, Nver=Nver, 
                            showhoverdata=showhoverdata, dipolesizeref=dipolesizeref, 
                            opacity=opacity, showlegend=showlegend,
                            sensorsize=sensorsize, sensoraxis=sensoraxis, streamlines=streamlines,
                            **kwargs)
            traces.extend(tcs)
        else:
            if sensorsources=='from_input':
                sensorsources = [obj for obj in input_objs if isSource(obj)]
            trace = getTrace(s, sensorsources=sensorsources, cst=cst, color=color, Nver=Nver, 
                            showhoverdata=showhoverdata, dipolesizeref=dipolesizeref, 
                            opacity=opacity, showlegend=showlegend,
                            sensorsize=sensorsize, sensoraxis=sensoraxis, streamlines=streamlines,
                            **kwargs)
            traces.append(trace)
    return traces

def getTrace(input_obj, sensorsources=None, cst=0, color=None, Nver=40, 
             showhoverdata=True, dipolesizeref=DIPOLESIZEREF, 
             opacity='default', showlegend=True, 
             sensorsize=SENSORSIZE, sensoraxis='z', streamlines=False, **kwargs):
    add_hover = ''
    s = input_obj
    kwargs['showlegend'] = showlegend
    kwargs['color'] = color
    try:
        kwargs['name'] = s.name
    except:
        pass
    if opacity == 'default':
        opacity = DISCRETESOURCE_OPACITY if isinstance(s, DiscreteSourceBox) else 1
    if isinstance(s, Box):
        if isinstance(s, DiscreteSourceBox):
            kwargs.pop('color')
            trace = makeDiscreteBox(data = s.data_downsampled,
                        pos=s.position, angle=s.angle, axis=s.axis, opacity=opacity, **kwargs)
        else:
            trace = makeBox(mag=s.magnetization, dim=s.dimension, 
                        pos=s.position, angle=s.angle, axis=s.axis, 
                        cst=cst, opacity=opacity,
                        **kwargs)
    elif isinstance(s, Cylinder):
        trace = makeCylinder(mag=s.magnetization, dim=s.dimension, 
                             pos=s.position, angle=s.angle,axis=s.axis,
                             N=Nver,
                             cst=cst, opacity=opacity,
                             **kwargs)
    elif isinstance(s, Sphere):
        trace = makeSphere(mag=s.magnetization, dim=s.dimension, 
                           pos=s.position, angle=s.angle, axis=s.axis, 
                           cst=cst, opacity=opacity, 
                           N=Nver,
                           **kwargs)
    elif isinstance(s, Line):
        trace = makeLine(curr=s.current,
                         vertices=s.vertices,
                         pos=s.position, angle=s.angle, axis=s.axis, 
                         opacity=opacity, 
                         **kwargs)
    elif isinstance(s, Circular):
        trace = makeCircular(curr=s.current,
                             dim=s.dimension,
                             pos=s.position, angle=s.angle, axis=s.axis, 
                             N=Nver, 
                             opacity=opacity,
                             **kwargs)
    elif isinstance(s, Dipole):
        trace = makeDipole(moment=s.moment,
                           pos=s.position, angle=s.angle, axis=s.axis, 
                           sizeref=dipolesizeref, 
                           opacity=opacity, 
                           **kwargs)
    elif isinstance(s, Sensor):
        if hasattr(s,'dimension'):
            sensorsize = s.dimension
        else:
            pass
        trace = makeSensor(pos=s.position, angle=s.angle, axis=s.axis, 
                           dim=sensorsize, 
                           opacity=opacity, 
                           **kwargs)
    elif isinstance(s, SurfaceSensor):
        n = '\n'
        if isinstance(s, Streamlines):
            trace = make_Streamlines(s, sensorsources=sensorsources, **kwargs)
            add_hover = f'''sources: {n.join([str(s).split(n)[0] for s in sensorsources])}'''
        elif isinstance(s, Surface):
            trace = make_Surface(s, sensorsources=sensorsources, sensoraxis=sensoraxis, showscale=False, **kwargs)
            add_hover = f'''sources: {n.join([str(s).split(n)[0] for s in sensorsources])}'''
        else:
            if 'name' not in kwargs:
                kwargs['name'] = 'Surface Sensor'
            trace = makeSensor(pos=s.position, angle=s.angle, axis=s.axis, 
                               dim=s.dimension, 
                               opacity=opacity, 
                               **kwargs)
    elif isinstance(s,RotationAxis):
        trace = make_RotationAxis(s, **kwargs)
    else:
        trace =  None
    
    if showhoverdata and trace is not None and not isinstance(s,RotationAxis):
        trace.hoverinfo = 'text'
        try:
            name = s.name + '<br>'
        except:
            name = ''
        trace.text = name + str(s).replace('\n', '<br>') + '<br>' + add_hover
    
    return trace

def displaySystem(*objs, figwidget=False, traces_properties=None, fig_layout=None):
    if traces_properties is None:
        traces_properties = {}
    if fig_layout is None:
        fig_layout = {}
    fig = go.Figure()
    fig.layout.scene.aspectmode = 'data'
    fig.add_traces(getTraces(*objs, **traces_properties))
    fig.update_layout(**fig_layout)
    if figwidget:
        return go.FigureWidget(fig)
    else:
        fig.show()


# %% [markdown]
# # Testing

# %%
box = Box(mag=(1,0,1), dim=(12, 10 ,12), pos=(0,0,0))
cylinder = Cylinder(mag=(0,1,0), dim=(13, 7), pos=(15,0,0))
sphere = Sphere(mag=(1,1,1), dim=11, pos=(30,0,0))
line = Line(curr=10, vertices=[(0,-4,0),(0,4,0)], axis=(1,0,0), angle= 45, pos=(0,0,15))
circular = Circular(curr=-5, dim=10,  pos=(10,0,15), axis=(1,0,0), angle=90)
dipole = Dipole(moment=(10,1,1), pos=(18,0,15))
discrete_source = DiscreteSourceBox('data/discrete_source_data.csv', pos=(25,0,15))
coll = Collection(box,cylinder,sphere, line, circular, dipole, discrete_source)

streamlines = Streamlines(Nelem=(5,5), dim=(8,8), pos=(0,0,15), angle=-45, axis=(1,0,0))
surface = Surface(Nelem=(5,5), dim=(8,8), pos=(0,0,15), angle=-45, axis=(1,0,0))
sensor = Sensor(pos=(35,0,15))
scoll = SensorCollection(sensor, surface)
traces_properties = dict(cst=0.2, sensorsources=[line], sensoraxis='z')
displaySystem(coll, sensor, streamlines, surface, traces_properties=traces_properties)

# %% [raw]
# sensor1 = Sensor(pos=(-10,0,0))
# sensor2 = Sensor(pos=(10,0,0))
# sensor3 = Sensor(pos=(0,0,0))
# scoll = SensorCollection(sensor1, sensor2)
#
# scoll.position = (-10,0,0)
# scoll.angle = 45
# scoll.axis = (0,1,0)
#
# scoll.position = (30,0,0)
# fig  = displaySystem(scoll, sensor3, figwidget=True)
# fig.layout.scene=dict(xaxis_range=[-50,50],
#                       yaxis_range=[-50,50],
#                       zaxis_range=[-50,50])
# fig

# %% [raw]
# sensorsources = [Box(mag=(0,1,0), dim=(1, 1 ,1), pos=(0,0,0))]
# surfsens = SurfaceSensor(Nelem=(20,20), dim=(2,8), pos=(0,0,0), angle=45, axis=(1,0,0))
#
# fig = go.FigureWidget(layout_scene_aspectmode='data')
# fig.add_traces(getTrace(*sensorsources))
# fig.add_trace(getTrace(surfsens, sensoraxis='y', sensorsources=sensorsources))
# fig.add_trace(getTrace(surfsens, sensorsources=sensorsources, streamlines=True, density=1))
# fig
