# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Imports

import numpy as np
import magpylib as magpy
import plotly.graph_objects as go
from magpylib._lib.mathLib import angleAxisRotation
from magpylib._lib.classes.magnets import Box, Cylinder, Sphere
from magpylib._lib.classes.currents import Line, Circular
from magpylib._lib.classes.moments import Dipole
from magpylib._lib.classes.sensor import Sensor


# # Sources definitions

# +
def makeSensor(pos = (0,0,0), angle=0, axis=(0,0,1), color=None, dim=10, **kwargs):
    box = go.Mesh3d(
        i = np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j = np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k = np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        showscale=False,
        name='box'
    )
    dim = np.array([1,1,0.2])*dim
    dd = 0.8 # shape modifier 
    x = np.array([-1, -1, 1, 1, -dd*1, -dd*1, dd*1, dd*1])*0.5*dim[0]+pos[0]
    y = np.array([-1, 1, 1, -1, -dd*1, dd*1, dd*1, -dd*1])*0.5*dim[1]+pos[1]
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])*0.5*dim[2]+pos[2]
    points = np.array([x,y,z])
    
    if color is not None:
        box.color = color
    if angle!=0:
        points = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
    
    box.x , box.y, box.z = points
    box.update(**kwargs)
    return box


def makeBox(mag=(0,0,1),  dim = (10,10,10), pos = (0,0,0), angle=0, axis=(0,0,1), cst=0.1, color=None, **kwargs):
    box = go.Mesh3d(
        i = np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j = np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k = np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        showscale=False,
        name='box'
    )
    x = np.array([-1, -1, 1, 1, -1, -1, 1, 1])*0.5*dim[0]+pos[0]
    y = np.array([-1, 1, 1, -1, -1, 1, 1, -1])*0.5*dim[1]+pos[1]
    z = np.array([-1, -1, -1, -1, 1, 1, 1, 1])*0.5*dim[2]+pos[2]
    points = np.array([x,y,z])
    
    if cst is not False:
        box.colorscale = _getColorscale(cst)
        box.intensity = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    elif color is not None:
        box.color = color
    if angle!=0:
        points = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
    
    box.x , box.y, box.z = points
    box.update(**kwargs)
    return box

def makeCylinder(mag=(0,0,1), dim = (5,10,0), pos = (0,0,0), angle=0, axis=(0,0,1), cst=False, color=None, N=40, **kwargs):
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
    cylinder=go.Surface(x=x, y=y, z=z, showscale=False, name='cylinder')
    if cst is not False:
        cylinder.colorscale = _getColorscale(cst)
        cylinder.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    elif color!=None:
        cylinder.colorscale = [[0,color],[1,color]]
    if angle!=0:
        points = np.array([x.flatten(),y.flatten(),z.flatten()])
        xr,yr,zr = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        cylinder.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    cylinder.update(**kwargs)
    return cylinder

def makeSphere(mag=(0,0,1), dim = 5, pos = (0,0,0), angle=0, axis=(0,0,1), cst=False, color=None, N=40, **kwargs):
    r = min(dim/2,dim/2)
    s =   np.linspace(0, 2*np.pi, 2*N)
    t =   np.linspace(0, np.pi, N)
    tGrid, sGrid =   np.meshgrid(s, t)

    x = r * np.cos(sGrid) * np.sin(tGrid)  
    y = r * np.sin(sGrid) * np.sin(tGrid)  
    z = r * np.cos(tGrid)                

    x,y,z = x+pos[0], y+pos[1], z+pos[2]
    sphere=go.Surface(x=x, y=y, z=z, showscale=False, name='sphere')
    
    if cst is not False:
        sphere.colorscale = _getColorscale(cst)
        sphere.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    elif color!=None:
        sphere.colorscale = [[0,color],[1,color]]
    if angle!=0:
        points = np.array([x.flatten(),y.flatten(),z.flatten()])
        xr,yr,zr = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        sphere.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    sphere.update(**kwargs)
    return sphere

def makeDipole(moment=(0.0, 0.0, 1), pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), color=None, sizeref=1, **kwargs):
    x,y,z = np.array([[p] for p in pos])
    moment= np.array(moment)/np.linalg.norm(moment)
    
    if angle!=0:
        moment = angleAxisRotation(moment, angle, axis, anchor=pos)
     
    u,v,w = [[m] for m in moment]
    dipole = go.Cone(x=x,y=y,z=z, u=u,v=v,w=w, sizeref = sizeref, name=f'dipole ({moment})mT/mm^3', sizemode = 'absolute', showscale=False)    
    dipole.update(**kwargs)
    return dipole


def makeLine(curr=0.0, vertices=[(-1.0, 0.0, 0.0),(1.0,0.0,0.0)], pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), color=None, **kwargs):
    x,y,z  = (np.array(vertices) + np.array(pos)).T
    points = np.array([x,y,z])
    
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        
    lineCurrent = go.Scatter3d(x=x,y=y,z=z,
                              mode = 'lines', line_width=5, name=f'line current ({curr:.2f}A)')    
    lineCurrent.update(**kwargs)
    return lineCurrent


def makeCircular(curr=0.0, dim=1.0, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), color=None, N=40, **kwargs):
    t =   np.linspace(0, 2*  np.pi, N)
    x = dim/2 * np.cos(t) + pos[0]
    y = dim/2 * np.sin(t) + pos[1]
    z =   np.ones(N)*pos[2]
    points = np.array([x,y,z])
    
    if angle!=0:
        x,y,z = np.array([angleAxisRotation(p, angle, axis, anchor=pos) for p in points.T]).T
        
    circularCurrent = go.Scatter3d(x=x,y=y,z=z,
                              mode = 'lines', line_width=8, name=f'circular current ({curr:.2f}A)')    
    circularCurrent.update(**kwargs)
    return circularCurrent


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


def getTrace(input_obj, cst=0, color=None, Nver=40, showhoverdata=True, dipolesizeref=1, opacity=1, showlegend=True, **kwargs):
    s = input_obj
    kwargs['showlegend'] = showlegend
    if isinstance(s, Box):
        trace = makeBox(mag=s.magnetization, dim=s.dimension, pos=s.position, angle=s.angle, axis=s.axis, color=color, cst=cst, opacity=opacity, **kwargs)
    elif isinstance(s, Cylinder):
        trace = makeCylinder(mag=s.magnetization, dim=s.dimension, pos=s.position, angle=s.angle, axis=s.axis, color=color, N=Nver, cst=cst, opacity=opacity, **kwargs)
    elif isinstance(s, Sphere):
        trace = makeSphere(mag=s.magnetization, dim=s.dimension, pos=s.position, angle=s.angle, axis=s.axis, color=color, N=Nver, cst=cst, opacity=opacity, **kwargs)
    elif isinstance(s, Line):
        trace = makeLine(curr=s.current, vertices=s.vertices, pos=s.position, angle=s.angle, axis=s.axis, color=color, opacity=opacity, **kwargs)
    elif isinstance(s, Circular):
        trace = makeCircular(curr=s.current, dim=s.dimension, pos=s.position, angle=s.angle, axis=s.axis, N=Nver, color=color, opacity=opacity, **kwargs)
    elif isinstance(s, Dipole):
        trace = makeDipole(moment=s.moment, pos=s.position, angle=s.angle, axis=s.axis, sizeref=dipolesizeref, color=color, opacity=opacity, **kwargs)
    elif isinstance(s, Sensor):
        if hasattr(s,'dimension'):
            sensorsize = s.dimension
        else:
            sensorsize=10
        trace = makeSensor(pos=s.position, angle=s.angle, axis=s.axis, dim=sensorsize, color=color, opacity=opacity, **kwargs)
    else:
        trace =  None
    if showhoverdata and trace is not None:
        trace.hoverinfo = 'text'
        trace.text = str(s).replace('\n', '<br>')
    return trace
# -


