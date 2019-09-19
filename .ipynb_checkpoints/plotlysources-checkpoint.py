# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import plotly.graph_objects as go
import numpy as np


# +
def _plotlyBox(mag=(0,0,1), pos = (0,0,0), dim = (10,10,10), angle=0, axis=(0,0,1), cst=0.1, **kwargs):
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
    
    if cst is not False:
        box.colorscale = _getColorscale(cst)
        box.intensity = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
        
    if angle!=0:
        x,y,z = AxisRotation(points=np.array([x,y,z]).T, angle=angle, axis=axis, anchor=pos).T
    
    box.x , box.y, box.z = x,y,z
    box.update(**kwargs)
    return box

def _plotlyCylinder(mag=(0,0,1), pos = (0,0,0), dim = (5,10,0), angle=0, axis=(0,0,1), cst=False, N=40, **kwargs):
    dim=np.array(dim)
    if len(dim)==2:
        dim = np.array(list(dim[0:2]) + [0])
    elif len(dim) == 3 and dim[2]==0:
        dim[2] = 1e-5
    ri = min(dim[0]/2,dim[2]/2)
    ro = max(dim[0]/2,dim[2]/2)
    hmin, hmax = -dim[1]/2, dim[1]/2
     
    h = [hmin,hmin,hmax,hmax,hmin]
    s = np.linspace(0, 2 * np.pi, N)
    sa, ha = np.meshgrid(s, h)

    ro = dim[0]/2  ; ri = dim[2]/2
    x = ro * np.cos(sa)
    y = ro * np.sin(sa)
    z = ha

    x[0] = x[-2] = x[-1] = ri*np.cos(s)
    y[0] = y[-2] = y[-1] = ri*np.sin(s)
    x,y,z = x+pos[0], y+pos[1], z+pos[2]
    cylinder=go.Surface(x=x, y=y, z=z, showscale=False, name='cylinder')
    if cst is not False:
        cylinder.colorscale = _getColorscale(cst)
        cylinder.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    if angle!=0:
        xr,yr,zr = AxisRotation(points=np.array([x.flatten(),y.flatten(),z.flatten()]).T, angle=angle, axis=axis, anchor=pos).T
        cylinder.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    cylinder.update(**kwargs)
    return cylinder

def _plotlySphere(mag=(0,0,1), pos = (0,0,0), dim = 5, angle=0, axis=(0,0,1), cst=False, N=40, **kwargs):
    r = min(dim/2,dim/2)
    s = np.linspace(0, 2 * np.pi, 2*N)
    t = np.linspace(0, np.pi, N)
    tGrid, sGrid = np.meshgrid(s, t)

    x = r * np.cos(sGrid) * np.sin(tGrid)  
    y = r * np.sin(sGrid) * np.sin(tGrid)  
    z = r * np.cos(tGrid)                

    x,y,z = x+pos[0], y+pos[1], z+pos[2]
    sphere=go.Surface(x=x, y=y, z=z, showscale=False, name='sphere')
    
    if cst is not False:
        sphere.colorscale = _getColorscale(cst)
        sphere.surfacecolor = _getIntensity(points=(x,y,z), mag=mag, pos=pos)
    if angle!=0:
        xr,yr,zr = AxisRotation(points=np.array([x.flatten(),y.flatten(),z.flatten()]).T, angle=angle, axis=axis, anchor=pos).T
        sphere.update(x=xr.reshape(x.shape), y=yr.reshape(y.shape), z=zr.reshape(z.shape))
    
    sphere.update(**kwargs)
    return sphere

def _plotlyLineCurrent(curr=0.0, vertices=[(-1.0, 0.0, 0.0),(1.0,0.0,0.0)], pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), **kwargs):
    x,y,z  = (np.array(vertices) + np.array(pos)).T
    
    if angle!=0:
        x,y,z = AxisRotation(points=np.array([x,y,z]).T, angle=angle, axis=axis, anchor=pos).T
        
    lineCurrent = go.Scatter3d(x=x,y=y,z=z,
                              mode = 'lines', line_width=5, name=f'line current ({curr:.2f}A)')    
    lineCurrent.update(**kwargs)
    return lineCurrent

def _plotlyDipole(moment=(0.0, 0.0, 1), pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), **kwargs):
    x,y,z = np.array([[p] for p in pos])
    moment= np.array(moment)/np.linalg.norm(moment)
    u,v,w =np.array([[m] for m in moment]) 
    
    if angle!=0:
        u,v,w = AxisRotation(points=np.array([u,v,w]).T, angle=angle, axis=axis, anchor=(0,0,0)).T
        
    dipole = go.Cone(x=x,y=y,z=z, u=u,v=v,w=w, sizeref = 1, name=f'dipole ({moment})mT/mm^3', sizemode = 'absolute', showscale=False)    
    dipole.update(**kwargs)
    return dipole

def _plotlyCircularCurrent(curr=0.0, dim=1.0, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), N=40, **kwargs):
    t = np.linspace(0, 2*np.pi, N)
    x = dim/2 * np.cos(t) + pos[0]
    y = dim/2 * np.sin(t) + pos[1]
    z = np.ones(N)*pos[2]
    
    if angle!=0:
        x,y,z = AxisRotation(points=np.array([x,y,z]).T, angle=angle, axis=axis, anchor=pos).T
        
    circularCurrent = go.Scatter3d(x=x,y=y,z=z,
                              mode = 'lines', line_width=5, name=f'circular current ({curr:.2f}A)')    
    circularCurrent.update(**kwargs)
    return circularCurrent
    
def AxisRotation(points,angle,axis,anchor):
    from scipy.spatial.transform import Rotation
    points= np.array(points)
    rotation = Rotation.from_rotvec(np.deg2rad(angle)*np.array(axis))  
    box_rotated = rotation.apply(points-anchor) + anchor
    return box_rotated

def _getIntensity(points, mag, pos):
    '''points: [x,y,z] array'''
    p = np.array(points)
    pos = np.array(pos)
    m = np.array(mag) / np.linalg.norm(mag)
    a = ((p[0]-pos[0])*m[0] + (p[1]-pos[1])*m[1] + (p[2]-pos[2])*m[2])
    b = (p[0]-pos[0])**2 + (p[1]-pos[1])**2 + (p[2]-pos[2])**2
    return a / np.sqrt(b)

def _getColorscale(cst=0.1):
    return [[0, 'turquoise'], [0.5*(1-cst), 'turquoise'],[0.5*(1+cst), 'magenta'], [1, 'magenta']]

# + {"active": ""}
# fig = go.FigureWidget()
# fig.add_trace(_plotlyBox(mag=(0,0,-1),dim=(40,40,20), pos=(-40,0,0), angle=45, axis=(1,1,1), cst=0.1))
# fig.add_trace(_plotlyCylinder(mag=(1,0,0),dim=(40,30), pos=(40,0,40), cst=0.1))
# fig.add_trace(_plotlySphere(mag=(0,1,0), dim=30, pos=(40,0,-40), cst=0.1))
# fig.add_trace(_plotlyLineCurrent(curr = 1, vertices = [(-30, 0.0, 30), (30, 0, 0)], pos=(10,-40,40), angle=-45))
# fig.add_trace(_plotlyCircularCurrent(curr=3, dim=50, angle=45, axis=(1,1,0), pos=(0,40,-50)))
# fig.add_trace(_plotlyDipole(moment=(1,2,-3), pos=(0,1,0), sizeref=20))
# fig
