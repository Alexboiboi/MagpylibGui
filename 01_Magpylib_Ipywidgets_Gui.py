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

# %% [markdown]
# # Imports

# %%
import plotly.graph_objects as go
import ipywidgets as widgets
from plotlytraces import getTrace, angleAxisRotation
import numpy as np
import magpylib as magpy
import plotly.figure_factory as ff
import plotly.io as pio
import glob
import os
import json

pio.templates['plotly_grey'] = pio.to_templated(go.Figure(layout_paper_bgcolor='rgb(33,33,33)', 
                                                          layout_plot_bgcolor='rgb(33,33,33)', 
                                                          layout_template='plotly_dark')).layout.template
default_theme = 'plotly_grey'
debug_view = widgets.Output()


# %% [markdown]
# # Streamlines

# %%
@debug_view.capture(clear_output=True, wait=True)
def update_streamlines(change=None):
        sk = streamlines_options
        test=[]
        for k,v in sk.items():
            test.append(v['checkbox'].value and v['density'].value>0.5)
        if sum(test)>0:
            continuous_update_checkbox.value = False
        
        for k,v in streamlines_options.items():
            if v['checkbox'].value == False:
                v['position'].layout.visibility = 'hidden'
                v['density'].layout.visibility = 'hidden'
            else:
                v['position'].layout.visibility = 'visible'
                v['density'].layout.visibility = 'visible'

        srx = np.array(figmag.layout.scene.xaxis.range)
        sry = np.array(figmag.layout.scene.yaxis.range)
        srz = np.array(figmag.layout.scene.zaxis.range)
        N=3
        if sk['xy']['checkbox'].value == True:
            sr = np.mean(np.diff([srx,sry]))
            pos = sk['xy']['position'].value
            density = sk['xy']['density'].value
            xs= np.linspace(srx[0],srx[1],int(N*density))+0.01
            ys= np.linspace(sry[0],sry[1],int(N*density))+0.01
            Bs = np.array([[src_col.getB([x,y,pos]) for x in xs] for y in ys])
            U,V = Bs[:,:,0], Bs[:,:,1]
            streamline = ff.create_streamline(x=xs, y=ys , u=U, v=V, density = 0.5*density, arrow_scale=sr*0.02)
            sl = streamline.data[0]
            t = figmag.data[1]
            t.mode = 'lines'
            with figmag.batch_update():
                t.visible = True
                t.x=sl.x ; t.y=sl.y ;  t.z=np.ones(len(sl.x))*pos
        else:
            figmag.data[1].visible = False

        if sk['xz']['checkbox'].value == True:
            sr = np.mean(np.diff([srx,srz]))
            pos = sk['xz']['position'].value
            density = sk['xz']['density'].value
            xs= np.linspace(srx[0],srx[1],int(N*density))+0.01
            zs= np.linspace(srz[0],srz[1],int(N*density))+0.01
            Bs = np.array([[src_col.getB([x,pos,z]) for x in xs] for z in zs])
            U,V = Bs[:,:,0], Bs[:,:,2]
            streamline = ff.create_streamline(x=xs, y=zs , u=U, v=V, density = 0.5*density, arrow_scale=sr*0.02)
            sl = streamline.data[0]
            t = figmag.data[2]
            t.mode = 'lines'
            with figmag.batch_update():
                t.visible = True
                t.x=sl.x ; t.y=np.ones(len(sl.x))*pos ;  t.z=sl.y
        else:
            figmag.data[2].visible = False

        if sk['yz']['checkbox'].value == True:
            sr = np.mean(np.diff([sry,srz]))
            pos = sk['yz']['position'].value
            density = sk['yz']['density'].value
            ys= np.linspace(sry[0],sry[1],int(N*density))+0.01
            zs= np.linspace(srz[0],srz[1],int(N*density))+0.01
            Bs = np.array([[src_col.getB([pos,y,z]) for y in ys] for z in zs])
            U,V = Bs[:,:,1], Bs[:,:,2]
            streamline = ff.create_streamline(x=ys, y=zs , u=U, v=V, density = 0.5*density, arrow_scale=sr*0.02)
            sl = streamline.data[0]
            t = figmag.data[3]
            t.mode = 'lines'
            with figmag.batch_update():
                t.visible = True
                t.x=np.ones(len(sl.x))*pos ; t.y=sl.x ;  t.z=sl.y
        else:
            figmag.data[3].visible = False

        
        
def _define_streamlines_widgets(density=1):
    streamlines_options={}
    for k,p in zip(['xy', 'xz', 'yz'],['z', 'y', 'x']):
        sk = streamlines_options[k] = dict()
        sk['checkbox'] = widgets.Checkbox(description=f'{k}-plane',  layout=dict(width='auto'), style=dict(description_width='0px'))
        sk['position'] = widgets.FloatSlider(description=f'{p}-position', min=-scene_range, max=scene_range, value=0, step=0.1, 
                                             continuous_update=False, layout=dict(flex='1'))
        sk['density'] = widgets.BoundedFloatText(description='density', min=1, max=5, step = 1, value=density, 
                                                 continuous_update=False, layout=dict(width='auto'), style=dict(description_width='auto'))
        sk['container']= widgets.HBox([sk['checkbox'], sk['position'], sk['density']], layout=dict(justify_content='space-between'))

        for s in sk.values():
            s.tag = k
            s.observe(update_streamlines, names='value')

    return streamlines_options,  widgets.VBox([v['container'] for v in streamlines_options.values()])


# %% [markdown]
# # Isosurface

# %%
@debug_view.capture(clear_output=True, wait=True)
def _define_isosurface_widgets(sr=100, density=20, opacity=0.5, surface_count=20):
    '''sr -> isosurface range'''
    density=20
    sr = 100
    isosurface_widgets_dict = dict(
        density_x = widgets.BoundedIntText(description='x', min=5, max=50, value=density, layout=dict(width='60px')),
        density_y = widgets.BoundedIntText(description='y', min=5, max=50, value=density, layout=dict(width='60px')),
        density_z = widgets.BoundedIntText(description='z', min=5, max=50, value=density, layout=dict(width='60px')),
        srx = widgets.FloatRangeSlider(description='x-range [mm]', min=-sr, max=sr, value=(-sr,sr), layout=dict(width='auto')),
        sry = widgets.FloatRangeSlider(description='y-range [mm]', min=-sr, max=sr, value=(-sr,sr), layout=dict(width='auto')),
        srz = widgets.FloatRangeSlider(description='z-range [mm]', min=-sr, max=sr, value=(-sr,sr), layout=dict(width='auto')),
        opacity = widgets.FloatSlider(description='opacity', min=0, max=1, step=0.1, value=opacity, layout=dict(width='auto')),
        surface_count = widgets.BoundedIntText(description='surf count', min=1, max=50, value=surface_count, layout=dict(width='110px')),
        isominmax = widgets.FloatRangeSlider(description='iso [%]', min=0, max=100, value=(10,100), layout=dict(width='auto'))
    )
    for w in isosurface_widgets_dict.values():
        w.style.description_width='auto'

    @debug_view.capture(clear_output=True, wait=True)
    def update_isosurface(srx=(-sr,sr), sry=(-sr,sr), srz=(-sr,sr), density_x=20, density_y=20, density_z=20, surface_count=10, opacity=0.5,
                         isominmax = (0,10)):
        for v in isosurface_widgets_dict.values():
            v.disabled = True
            v.layout.flex ='1'
        X, Y, Z = np.mgrid[srx[0]:srx[1]:density_x*1j, sry[0]:sry[1]:density_y*1j, srz[0]:srz[1]:density_z*1j]
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()

        Bs = np.array([np.linalg.norm(src_col.getB([x,y,z])) for x,y,z in zip(x,y,z)]).flatten()
        imin = isominmax[0]*(np.max(Bs)-np.min(Bs))*0.01
        imax = isominmax[1]*(np.max(Bs)-np.min(Bs))*0.01

        t = figmag.data[4]
        t.colorbar.title = 'magB [mT]'
        t.update(x=x, y=y, z=z, value=Bs, visible=True, surface_count=surface_count, opacity = opacity, isomin=imin, isomax=imax)
        for v in isosurface_widgets_dict.values():
            v.disabled = False

    hide_button = widgets.Button(description='remove', button_style='warning', layout=dict(width='auto'))
    hide_button.on_click(lambda b: _clear_isosurface_data())
    iw = widgets.interactive(update_isosurface, {"manual":True, "manual_name":'update'}, **isosurface_widgets_dict)
    iw.manual_button.button_style='success'
    iw.manual_button.layout.width='auto'
    iwd = isosurface_widgets_dict
    isosurface_widgets = widgets.VBox([iwd['srx'], iwd['sry'], iwd['srz'], iwd['isominmax'], iwd['opacity'],
                                       widgets.HBox([widgets.HTML('density: '), 
                                                     iwd['density_x'], iwd['density_y'], iwd['density_z'], 
                                                     iwd['surface_count'], iw.manual_button, hide_button])
                                      ])
    return isosurface_widgets_dict, isosurface_widgets


# %% [markdown]
# # Update objects

# %% [markdown]
# ## Update Source

# %%
@debug_view.capture(clear_output=True, wait=True)
def update_source(source_id):
    source_type = source_id.split('_')[0]
    if source_type in ['box', 'cylinder', 'sphere']:
        update_magnet(source_id)
    if source_type == 'dipole':
        update_dipole(source_id)
    if source_type in ['line current', 'circular current']:
        update_current(source_id)

@debug_view.capture(clear_output=True, wait=True)
def on_source_change(change):
    update_source(change.owner.id)
        
@debug_view.capture(clear_output=True, wait=True)
def observe_source(source_id, val=True):
    for w in sources[source_id]['widgets'].values():
        if val:
            w.observe(on_source_change, names='value')
        else:
            w.unobserve(on_source_change, names='value')
    if val:
        update_source(source_id)
        
@debug_view.capture(clear_output=True, wait=True)
def update_magnet(source_id):
    global sources
    _clear_isosurface_data()
    sp = sources[source_id]
    w =sp['widgets']
    axis = (w['axis_x'].value, w['axis_y'].value, w['axis_z'].value)
    pos = (w['xpos'].value, w['ypos'].value, w['zpos'].value)
    angle = w['angle'].value
    sp['magpy_source'].setOrientation(angle, axis)
    sp['magpy_source'].setPosition(pos)
    sp['magpy_source'].magnetization = np.array([w['Mx'].value, w['My'].value, w['Mz'].value])
    
    if sp['source_type'] == 'box':
        sp['magpy_source'].dimension = np.array([w['Lx'].value, w['Ly'].value, w['Lz'].value])
    elif sp['source_type'] == 'cylinder':
        sp['magpy_source'].dimension = np.array([w['Lx'].value, w['Ly'].value])
    elif sp['source_type'] == 'sphere':
        sp['magpy_source'].dimension = w['Lx'].value
        
    if w['update3d'].value:
        tm = sp['trace']
        update_streamlines()

        with figmag.batch_update():
            tm.update(getTrace(sp['magpy_source'], name=sp['trace'].name))
    
    update_sensors_reading(parent = sp)
        
            
@debug_view.capture(clear_output=True, wait=True)
def update_current(source_id):
    global sources
    _clear_isosurface_data()
    sp = sources[source_id]
    w =sp['widgets']
    axis = (w['axis_x'].value, w['axis_y'].value, w['axis_z'].value)
    pos = (w['xpos'].value, w['ypos'].value, w['zpos'].value)
    angle = w['angle'].value
    curr = w['curr'].value

    sp['magpy_source'].setOrientation(angle, axis)
    sp['magpy_source'].setPosition(pos)
    
    if sources[source_id]['widgets']['update3d'].value == True:
        tm = sp['trace']
        update_streamlines()
        with figmag.batch_update():
            if sp['source_type'] == 'lineCurrent':
                vertices = sp['magpy_source'].vertices = [tuple(w[f'{n}{i}'].value for n in ['x','y','z']) for i,v in enumerate(sp['properties']['vertices'])]
                tm.update(getTrace(sp['magpy_source'], name=sp['trace'].name))
            elif sp['source_type'] == 'circularCurrent':
                dim = sp['magpy_source'].dimension = w['d'].value
                tm.update(getTrace(sp['magpy_source'], name=sp['trace'].name))
                
    update_sensors_reading(parent = sp)
    
@debug_view.capture(clear_output=True, wait=True)
def update_dipole(source_id):
    global sources
    _clear_isosurface_data()
    sp = sources[source_id]
    w =sp['widgets']
    axis = (w['axis_x'].value, w['axis_y'].value, w['axis_z'].value)
    pos = (w['xpos'].value, w['ypos'].value, w['zpos'].value)
    angle = w['angle'].value
    moment = (w['moment_x'].value, w['moment_y'].value, w['moment_z'].value)
    sp['magpy_source'].setOrientation(angle, axis)
    sp['magpy_source'].setPosition(pos)
    sp['magpy_source'].moment = moment
        
    if sources[source_id]['widgets']['update3d'].value == True:
        tm = sp['trace']
        update_streamlines()
        with figmag.batch_update():
            tm.update(getTrace(sp['magpy_source'], dipolesizeref=w['sizeref'].value, name=sp['trace'].name))
    
    update_sensors_reading(parent = sp)
    
@debug_view.capture(clear_output=True, wait=True)
def update_all_sources():
    for k,v in sources.items():
        v['widgets']['update3d'].value = False # first False to ensure that a change to True triggers all update functions
        v['widgets']['update3d'].value = True
        
@debug_view.capture(clear_output=True, wait=True)
def update_sources_titles(source_id=None):
    '''if None every source reading is updated'''
    if sources:
        if source_id is None:
            sd = sources
        else:
            sd = {'source_id': sources[source_id]}

        for k,sp in sd.items():
            nw = sp['name_widget'].value.strip()
            name = f"{sp['id']}" if nw=='' else f"{nw}"
            sp['trace'].name = nw
            ind = list(sources.keys()).index(sp['id'])
            if sp['update3d_checkbox'].value == False:
                title = "{} (3d update OFF)".format(name)
            else:
                title = "{} (3d update ON)".format(name)
            sources_list_accordion.set_title(ind, title)


# %% [markdown]
# ## Update Sensor

# %%
@debug_view.capture(clear_output=True, wait=True)
def on_sensor_change(change):
    update_sensor(change.owner.id)
    
@debug_view.capture(clear_output=True, wait=True)
def observe_sensor(sensor_id, val=True):
    for w in sensors[sensor_id]['widgets'].values():
        if val:
            w.observe(on_sensor_change, names='value')
        else:
            w.unobserve(on_sensor_change, names='value')
    if val:
        update_sensor(sensor_id)
        
@debug_view.capture(clear_output=True, wait=True)
def update_sensor(sensor_id):
    global sensors
    sp = sensors[sensor_id]
    w =sp['widgets']
    axis = (w['axis_x'].value, w['axis_y'].value, w['axis_z'].value)
    pos = (w['xpos'].value, w['ypos'].value, w['zpos'].value)
    angle = w['angle'].value
    sp['magpy_sensor'].setOrientation(angle, axis)
    sp['magpy_sensor'].setPosition(pos)

    magB = update_sensors_reading(sensor_id)
    rb = sp['record_B_button'].description
    
    if w['update3d'].value:
        if sp['sensor_type'] == 'hall-3d':
            tm = sp['trace']
        with figmag.batch_update():
            tm.update(getTrace(sp['magpy_sensor'], dimension=w['sensorsize'].value, name=sp['trace'].name))
            if rb != 'record B':
                x,y,z = pos
                bt = sensors[sensor_id]['Brecord_trace3d']
                bt.x+=(x,) ; bt.y+=(y,) ; bt.z+=(z,)
    
    if w['updateBplot'].value and rb != 'record B':
        if rb == 'stop recording B':
            rb = 'clear recording'
        fs = sensors[sensor_id]['fig_Brecord']
        with fs.batch_update():
            for i in range(3):
                if magB is not None:
                    fs.data[i].y+= (magB[i],)
        
@debug_view.capture(clear_output=True, wait=True)
def update_sensors_reading(sensor_id=None, parent=None):
    '''if None every sensor reading is updated'''
    if sensors:
        if sensor_id is None:
            sd = sensors
        else:
            sd = dict(sensor_id = sensors[sensor_id])

        for k,sp in sd.items():
            name = f"{sp['id']}" if sp['name_widget'].value.strip()=='' else f"{sp['name_widget'].value.strip()}"
            ind = list(sensors.keys()).index(sp['id'])
            if sources:
                magB = sp['magpy_sensor'].getB(*src_col.sources)
                title = "{} (Bx={:.3f}mT, By={:.3f}mT, Bz={:.3f}mT)".format(name, *magB)
                sensors_list_accordion.set_title(ind, title)
                if parent is not None and sp['tabs'].selected_index==2:
                    if parent['tabs'].selected_index!=4:
                        u = sp['circle_widgets']['updateBplot'] #updatibg B plot indirectly
                        u.value = False
                        u.value = True
                return magB
            else:
                title = "{} (no sources)".format(name)
                sensors_list_accordion.set_title(ind, title)
            
@debug_view.capture(clear_output=True, wait=True)
def update_all_sensors():
    for k,v in sensors.items():
        v['widgets']['update3d'].value = False # first False to ensure that a change to True triggers all update functions
        v['widgets']['update3d'].value = True


# %% [markdown]
# ## Update genral

# %%
@debug_view.capture(clear_output=True, wait=True)
def on_continuous_update_change(change):
    for s in sources.values():
        for v in s['widgets'].values():
            v.continuous_update = change.owner.value
    for s in sensors.values():
        for v in s['widgets'].values():
            v.continuous_update = change.owner.value
        

@debug_view.capture(clear_output=True, wait=True)
def update_objects(*ids, target_group = None):
    if not ids:
        if target_group == 'sensors':
            update_all_sensors()
        elif target_group == 'sources':
            update_all_sources()
        elif target_group == 'all':
            update_all_sensors()
            update_all_sources()
    elif ids and target_group is None:
        for i in ids:
            if i in sensors:
                update_sensor(id)
            elif i in sources:
                update_source(id)


# %% [markdown]
# # Object rotation

# %%
@debug_view.capture(clear_output=True, wait=True)
def record_object_rotation(object_id, angle_start=0, angle_end=360, angle_step=1, axis=(0,0,1), anchor=(0,0,0)):
    if object_id in sources:
        sp = sources[object_id]
        magpy_obj = sp['magpy_source']
        S = sensors
    else:
        sp = sensors[object_id]
        magpy_obj = sp['magpy_sensor']
        S = dict(object_id=sp)
    
    for s in S.values():
        B=[]
        angles = np.arange(angle_start,angle_end,angle_step)
        pos = tuple(s['magpy_sensor'].position)
        for a in angles:
            magpy_obj.rotate(angle=angle_step , axis=axis, anchor=anchor)
            B.append(s['magpy_sensor'].getB(*src_col.sources))
        B = np.array(B).T
        for i,t in enumerate(s['fig_B_circle'].data):
            t.y = B[i]
            t.x = angles
        obj_name = sp['name_widget'].value
        sens_name = s['name_widget'].value
        s['fig_B_circle'].layout.xaxis.title = f'{obj_name} angle [deg]'
        s['fig_B_circle'].layout.yaxis.title = f'B [mT] - {sens_name}'
        s['fig_B_circle'].layout.title.text = f'{obj_name} rotation  with anchor:{np.array(anchor).round(2)},  ' + \
                                     f'axis:{np.array(axis).round(2)},  ' + \
                                     f'start position:{np.array(pos).round(2)}'

#record_object_rotation('hall-3d_01', angle_start=0, angle_end=360, angle_step=1, axis=(0,0,1), anchor=(0,0,0))


# %%
@debug_view.capture(clear_output=True, wait=True)
def define_rotation_widgets(object_id, axis=(0,0,1), anchor=(0,0,0)):
    if object_id in sources:
        sp = sources[object_id]
    elif object_id in sensors:
        sp = sensors[object_id]
        
    wwidth = '100px'
    circle_widgets = cw = dict(
        xpos = widgets.FloatText(description='x', layout=dict(width=wwidth)),
        ypos = widgets.FloatText(description='y', layout=dict(width=wwidth)),
        zpos = widgets.FloatText(description='z', layout=dict(width=wwidth)),
        angle_range = widgets.FloatRangeSlider(description='arc range [deg]', min=0, max=360, value=[0,360],
                                               layout=dict(flex='1'), style=dict(handle_color='green')),
        angle_step = widgets.BoundedFloatText(description='step', min=0.1, max=20, value=10, layout=dict(width=wwidth)),
        axis_x= widgets.FloatText(description='x', min=-1, max=1, step=0.1, value=axis[0], layout=dict(width=wwidth)),
        axis_y= widgets.FloatText(description='y', min=-1, max=1, step=0.1, value=axis[1], layout=dict(width=wwidth)),
        axis_z= widgets.FloatText(description='z', min=-1, max=1, step=0.1, value=axis[2], layout=dict(width=wwidth)),
        anchor_x= widgets.FloatText(description='x', min=-100, max=100, step=0.1, value=anchor[0], layout=dict(width=wwidth)),
        anchor_y= widgets.FloatText(description='y', min=-100, max=100, step=0.1, value=anchor[1], layout=dict(width=wwidth)),
        anchor_z= widgets.FloatText(description='z', min=-100, max=100, step=0.1, value=anchor[2], layout=dict(width=wwidth)),
        updateBplot = widgets.Checkbox(description='update B plot', value=False, style=dict(description_width='auto')),
        update_circle = widgets.Checkbox(description='update circle', value=False, style=dict(description_width='auto'))
    )
    
    widgets.jsdlink((cw['update_circle'], 'value'), (cw['updateBplot'], 'value'))
    widgets.jslink((cw['angle_step'], 'value'), (cw['angle_range'], 'step'))
    
    for w in circle_widgets.values():
        w.style.description_width='auto'
        w.continuous_update = False

    @debug_view.capture(clear_output=True, wait=True)
    def update_arc(xpos, ypos, zpos, angle_range, angle_step, axis_x, axis_y, axis_z, anchor_x, anchor_y, anchor_z, updateBplot, update_circle):
        if object_id in sources:
            sp = sources[object_id]
        elif object_id in sensors:
            sp = sensors[object_id]
            
        axis=(axis_x, axis_y, axis_z)
        anchor=(anchor_x, anchor_y, anchor_z)
            
        if update_circle:
            pos_start=(xpos,ypos,zpos)
            angles = np.arange(angle_range[0], angle_range[1], angle_step)
            name = f'{object_id} rotation ({len(angles)}points)'
            x,y,z = np.array([angleAxisRotation(pos_start, a, axis, anchor=anchor) for a in angles]).T    

            with figmag.batch_update():
                sp['circle_array_trace'].update(x=x, y=y, z=z, name=name)
        else:
            sp['circle_array_trace'].update(x=[], y=[], z=[])
            
        if updateBplot:
            record_object_rotation(object_id, angle_start=angle_range[0], angle_end=angle_range[1], 
                                   angle_step=angle_step, axis=axis, anchor=anchor)
  
    widgets.interactive(update_arc, **circle_widgets)
    
    
    rotation_widgets_ui = widgets.VBox([
        widgets.HBox([cw['angle_range'], cw['angle_step']]),
        widgets.HBox([widgets.HTML('start position: '), widgets.HBox([cw['xpos'], cw['ypos'], cw['zpos']])],
                                                                    layout=dict(justify_content='space-between')),
        widgets.HBox([widgets.HTML('circle axis: '), widgets.HBox([cw['axis_x'], cw['axis_y'], cw['axis_z']])],
                                                                 layout=dict(justify_content='space-between')),
        widgets.HBox([widgets.HTML('rotation anchor: '), widgets.HBox([cw['anchor_x'], cw['anchor_y'], cw['anchor_z']])],
                                                                     layout=dict(justify_content='space-between')),
        #widgets.HBox([cw['updateBplot'], cw['update_circle']],layout=dict(justify_content='space-between')),
    ])
    return cw, rotation_widgets_ui


# %% [markdown]
# # Add objects

# %% [markdown]
# ## Add source

# %%
@debug_view.capture(clear_output=True, wait=True)
def add_source(source_type='box', mag=(0,0,100),dim=(50,50,50), pos=(0,0,0), angle=0, axis=(0,0,1), curr=1, 
               vertices=[(-10,0,0),(10,0,0)], moment=(1,0,0), dipolesizeref=10, name=None):
    
    _clear_isosurface_data()
    graphics_container.children = [graphics_window, streamlines_accordion, isosurface_accordion]
    
    source_props  = {'pos':pos, 'angle':angle, 'axis':axis}
    for i in range(1,100):
        source_id = f'{source_type}_{i:02d}'
        if source_id not in sources.keys():
            break
            
    sources[source_id] = {'id': source_id}
                          
    if name is None:
        name = source_id
        
    delete_source_button = widgets.Button(description='delete', icon='trash', button_style='danger',
                                          layout=dict(width='auto'))
    delete_source_button.on_click(on_delete_source_button_click)
    delete_source_button.id = source_id
    
    source_name_widget = widgets.Text(description='new source name', value = name, 
                                      style=dict(description_width='auto'))
    source_name_widget.id = source_id
    def update_source_title(change):
        update_sources_titles(source_id)
    source_name_widget.observe(update_source_title, names = 'value')
    
    def rename_source(b):
        if b.icon=='check':
            source_name_container.children =[rename_source_button, source_opacity_slider, delete_source_button]
        else:
            source_name_container.children =[widgets.HBox([source_name_widget, ok_button])]
    rename_source_button = widgets.Button(description='rename', icon='text', button_style='warning',layout=dict(width='auto'))
    rename_source_button.on_click(rename_source)
    ok_button = widgets.Button(icon='check', button_style='success',layout=dict(width='auto'))
    ok_button.on_click(rename_source)
    
    def update_source_opacity(opacity):
        trace.opacity = opacity
    source_opacity_slider = widgets.FloatSlider(description='opacity', value=1, min=0, max=1)
    widgets.jsdlink((opacity_slider, 'value'), (source_opacity_slider, 'value'))
    widgets.interactive(update_source_opacity, opacity=source_opacity_slider)
    
    source_name_container = widgets.HBox([rename_source_button, source_opacity_slider, delete_source_button],
                                         layout=dict(justify_content='space-between'))
    
    
    update3d_checkbox = widgets.Checkbox(value=True, description='update 3d plot',
                                         style=dict(description_width='0px'), layout=dict(width='auto'))
    widgets.jsdlink((update3d_all_updates_checkbox,'value'), (update3d_checkbox, 'value'))
    @debug_view.capture(clear_output=True, wait=True)
    def on_update3d_change(change):
        update_sources_titles(source_id)
    update3d_checkbox.observe(on_update3d_change, names='value')
    
    cst = 0 #color scale threshold
    
    
    orientation_widgets = dict(
        angle = widgets.FloatSlider(description='angle [deg]', min=-180, max=180, value=angle, style=dict(handle_color='blue')),
        axis_x= widgets.FloatSlider(description='x', min=-1, max=1, step=0.1, value=axis[0]),
        axis_y= widgets.FloatSlider(description='y', min=-1, max=1, step=0.1, value=axis[1]),
        axis_z= widgets.FloatSlider(description='z', min=-1, max=1, step=0.1, value=axis[2])
    )
    position_widgets = dict(
        xpos = widgets.FloatSlider(description='x [mm]', min=pos[0]-100, max=pos[0]+100, step=0.1, value=pos[0]),
        ypos = widgets.FloatSlider(description='y [mm]', min=pos[1]-100, max=pos[1]+100, step=0.1, value=pos[1]),
        zpos = widgets.FloatSlider(description='z [mm]', min=pos[2]-100, max=pos[2]+100, step=0.1, value=pos[2])
    )
    
    all_widgets_source = dict(**orientation_widgets, **position_widgets)
        
    orient_buttons={}
    @debug_view.capture(clear_output=True, wait=True)
    def _on_orient_button_click(b):
        vals = {'angle':0, 'axis_x':0, 'axis_y':0, 'axis_z':0}
        if b.description == 'x':
            vals['axis_y'] = 1 ; vals['angle']=90
        elif b.description == 'y':
            vals['axis_x'] = 1 ; vals['angle']=90
        elif b.description == 'z':
            vals['axis_z'] = 1
        observe_source(source_id, val=False)
        for k,v in orientation_widgets.items():
            v.value = vals[k]
        observe_source(source_id, val=True)
            
    for o in ['x','y','z']:
        ob = widgets.Button(description=o, icon='check', tooltip=f'orient along {o}-axis', layout=dict(width='120px'))
        ob.on_click(_on_orient_button_click)
        orient_buttons[f'axis_{o}'] = ob
    orient_button_HBox = widgets.HBox(list(orient_buttons.values()), layout=dict(justify_content='space-between'))
    
    up_widgs = widgets.HBox([continuous_update_checkbox, update3d_checkbox], layout=dict(justify_content='space-between'))
    
    cw, rotation_widgets_ui = define_rotation_widgets(source_id)
    
    tabs = widgets.Tab([widgets.VBox(list(position_widgets.values()) + [up_widgs]),
                        widgets.VBox([orient_button_HBox] + list(orientation_widgets.values()) + [up_widgs])
                       ])
    
    tabs.set_title(0,'position')
    tabs.set_title(1,'orientation')
       
    if source_type in ['box', 'sphere', 'cylinder']:
        magnetization_widgets = dict(
            Mx = widgets.FloatText(description='x [mT]', step=0.1, value=mag[0]),
            My = widgets.FloatText(description='y [mT]', step=0.1, value=mag[1]),
            Mz = widgets.FloatText(description='z [mT]', step=0.1, value=mag[2])
        )
        if source_type=='box':
            source_props.update(**{'mag':mag, 'dim':dim})
            magpy_source = magpy.source.magnet.Box(**source_props)
            figmag.add_trace(getTrace(magpy_source, cst=cst, name=source_name_widget.value))
            dimensions_widgets = dict(
                Lx = widgets.FloatSlider(description='x [mm]', min=0, max=dim[0]*5, step=0.1, value=dim[0]),
                Ly = widgets.FloatSlider(description='y [mm]', min=0, max=dim[1]*5, step=0.1, value=dim[1]),
                Lz = widgets.FloatSlider(description='z [mm]', min=0, max=dim[2]*5, step=0.1, value=dim[2])
                                     )
        elif source_type=='cylinder':
            source_props.update(**{'mag':mag, 'dim':dim})
            magpy_source = magpy.source.magnet.Cylinder(**source_props)
            if len(dim)==2:
                dim = list(dim) + [dim[0]]
            figmag.add_trace(getTrace(magpy_source, cst=cst, name=source_name_widget.value))
            dimensions_widgets = dict(
                Lx = widgets.FloatSlider(description='d_outer [mm]', min=0, max=dim[0]*5, step=0.1, value=dim[0]),
                Ly = widgets.FloatSlider(description='h [mm]', min=0, max=dim[1]*5, step=0.1, value=dim[1]),
                #Lz = widgets.FloatSlider(description='d_inner [mm]', min=0, max=dim[0]*5, step=0.1, value=dim[2])
            )
        elif source_type=='sphere':
            source_props.update(**{'mag':mag, 'dim':dim})
            magpy_source = magpy.source.magnet.Sphere(**source_props)
            figmag.add_trace(getTrace(magpy_source, cst=cst, name=source_name_widget.value))
            dimensions_widgets = dict(
                Lx = widgets.FloatSlider(description='r [mm]', min=0, max=dim*5, step=0.1, value=dim),
            )
            
        def on_scale_change(s):
            observe_source(source_id, val=False)
            for w in dimensions_widgets.values():
                w.value*=s
            observe_source(source_id, val=True)
        scale_widget = widgets.BoundedFloatText(min=0.1, max=10, value=1, step=0.1, description='scale',
                                               layout=dict(width='150px'))
        sw = widgets.interactive(on_scale_change, {"manual":True, "manual_name":'ok'}, s=scale_widget)
        sw.manual_button.button_style='success'
        sw.manual_button.layout.width='auto'
        all_widgets_source.update(**magnetization_widgets, **dimensions_widgets, update3d=update3d_checkbox)
        tabs.children += (widgets.VBox(list(dimensions_widgets.values()) + [widgets.HBox([scale_widget, sw.manual_button]), up_widgs]), 
                        widgets.VBox(list(magnetization_widgets.values()) + [up_widgs])
                       )
        tabs.set_title(2,'dimensions')
        tabs.set_title(3,'magnetization')
    
    elif source_type in ['lineCurrent', 'circularCurrent']:
        C = dict(curr = widgets.FloatSlider(description='current [A]', min=-10*curr, max=10*curr, step=0.1, value=curr))
        all_widgets_source.update(**C)
        if source_type == 'lineCurrent':
            source_props.update(**{'curr':curr, 'vertices':vertices})
            magpy_source = magpy.source.current.Line(curr, vertices, pos, angle, axis)
            figmag.add_trace(getTrace(magpy_source, cst=cst, name=source_name_widget.value))
            v_dict = dict()
            for i,ver in enumerate(vertices):
                for k,v in zip([f'{n}{i}' for n in ['x','y','z']], ver):
                        v_dict[k] = widgets.FloatSlider(description=f'{k} [mm]', min=-100, max=100, step=0.1, value=v)
                        v_dict[k].id = source_id
            all_widgets_source.update(v_dict, update3d=update3d_checkbox)
            tabs.children += (widgets.VBox(list(v_dict.values()) + [up_widgs]),)
            tabs.set_title(2,'vertices')
        else:
            source_props.update(**{'curr':dim, 'dim':dim})
            magpy_source = magpy.source.current.Circular(curr, dim, pos, angle, axis)
            figmag.add_trace(getTrace(magpy_source, cst=cst, name=source_name_widget.value))
            D = dict(d = widgets.FloatSlider(description='d [mm]', min=-10*dim, max=10*dim, step=0.1, value=dim))
            all_widgets_source.update(**D, update3d=update3d_checkbox)
            tabs.children += (widgets.VBox(list(D.values()) + [up_widgs]),)
            tabs.set_title(2,'dimension')
        
        tabs.children += (widgets.VBox(list(C.values()) + [up_widgs]),)
        tabs.set_title(3,'current')
    elif source_type == 'dipole':
        source_props.update(**{'moment':moment})  
        magpy_source = magpy.source.moment.Dipole(moment=moment, pos=pos, angle=angle, axis=axis)
        figmag.add_trace(getTrace(magpy_source, dipolesizeref=dipolesizeref, cst=cst, name=source_name_widget.value))
        M = dict(moment_x = widgets.FloatSlider(description='x [mT*mm^3]', min=-10, max=10, step=0.1, value=moment[0]),
                 moment_y = widgets.FloatSlider(description='y [mT*mm^3]', min=-10, max=10, step=0.1, value=moment[1]),
                 moment_z = widgets.FloatSlider(description='z [mT*mm^3]', min=-10, max=10, step=0.1, value=moment[2]),
                 sizeref = widgets.FloatSlider(description='sizeref', min=0, max=100, step=0.1, value=dipolesizeref),
                )
        all_widgets_source.update(**M, update3d=update3d_checkbox)
        tabs.children += (widgets.VBox(list(M.values()) + [up_widgs]),)
        tabs.set_title(2,'moment')
    
    trace = figmag.data[-1]
    
    for w in all_widgets_source.values():
        w.style.description_width='auto'
        w.layout.width='auto'
        w.continuous_update = False
    
    figB_circle_container = widgets.VBox()
    tabs.children += (widgets.VBox(list(rotation_widgets_ui.children) + [figB_circle_container]),)
    tabs.set_title(len(tabs.children)-1,'rotation array')
                 
    src_col.addSources(magpy_source)
                      
    figmag.add_scatter3d(marker_size=2)
    circle_array_trace = figmag.data[-1] 
                      
    sources[source_id].update(**{
      'source_type':source_type, 
      'name_widget':source_name_widget, 
      'trace': trace,
      'traces': [trace, circle_array_trace],
      'circle_array_trace': circle_array_trace,
      'rotation_widgets_ui': rotation_widgets_ui,
      'properties':source_props, 
      'magpy_source': magpy_source,
      'update3d_checkbox': update3d_checkbox,
      'widgets': all_widgets_source,
      'tabs':tabs,
      'widget': widgets.VBox([source_name_container,
                             tabs])
    })
    
    sw = sources[source_id]['widgets']
    for k in ['x','y','z']:
        widgets.jslink((sw[k+'pos'],'value'),(cw[k+'pos'],'value'))
    
    @debug_view.capture(clear_output=True, wait=True)
    def on_selected_tab_change(change):
        if change.new == 4:
            cw['update_circle'].value=True
            figB_circle_container.children = [s['fig_B_circle'] for s in sensors.values()]
            for s in sensors.values():
                s['tabs'].selected_index = 0
        else:
            cw['update_circle'].value=False
            figB_circle_container.children = []
        

    tabs.observe(on_selected_tab_change, names='selected_index')
    
    
    sources_list_accordion.children+=(sources[source_id]['widget'],)
    sources_list_accordion.set_title(len(sources_list_accordion.children)-1, f"{name}")
    sources_list_accordion.selected_index = len(sources_list_accordion.children)-1
    
    update_sensors_reading(parent = sources[source_id])
    update_sources_titles(source_id)
    
    for w in all_widgets_source.values():
        w.id = source_id
        #w.source_type = source_type
    observe_source(source_id, val=True)
    
    if len(sources)>1:
        delete_all_sources_button_container.children = [delete_all_sources_button]


# %% [markdown]
# ## Add sensor

# %%
@debug_view.capture(clear_output=True, wait=True)
def add_sensor(sensor_type='hall-3d', pos=(0,0,0), angle=0, axis=(0,0,1), name=None, sensorsize=16):
        
    sensor_props  = {'pos':pos, 'angle':angle, 'axis':axis}
    for i in range(1,100):
        sensor_id = f'{sensor_type}_{i:02d}'
        if sensor_id not in sensors.keys():
            break
    sensors[sensor_id] = {'id': sensor_id}
    
    if name is None:
        name = sensor_id
        
    delete_sensor_button = widgets.Button(description='delete', icon='trash', button_style='danger',
                                         layout=dict(width='auto'))
    delete_sensor_button.on_click(on_delete_sensor_button_click)
    delete_sensor_button.id = sensor_id
    def update_sensor_title(change):
        sensors_list_accordion.set_title(sensors_list_accordion.selected_index, f"{change.owner.id}" if change.new.strip()=='' else f"{change.new}")
    sensor_name_widget = widgets.Text(description='new sensor name', value = name, style=dict(description_width='auto'))
    sensor_name_widget.id = sensor_id
    sensor_name_widget.observe(update_sensor_title, names = 'value')
    
    update3d_checkbox = widgets.Checkbox(value=True, description='update 3d plot', style=dict(description_width='0px'), layout=dict(width='auto'))
    widgets.jsdlink((update3d_all_updates_checkbox,'value'), (update3d_checkbox, 'value'))
    
    if sensor_type=='hall-3d':
        magpy_sensor = magpy.Sensor(**sensor_props)
        figmag.add_trace(getTrace(magpy_sensor, name=sensor_name_widget.value))
        sensor_trace = figmag.data[-1]
    

        orientation_widgets = dict(angle = widgets.FloatSlider(description='angle [deg]', min=-180, max=180, value=angle, 
                                                               style=dict(handle_color='blue')),
                                   axis_x= widgets.FloatSlider(description='x', min=-1, max=1, step=0.1, value=axis[0]),
                                   axis_y= widgets.FloatSlider(description='y', min=-1, max=1, step=0.1, value=axis[1]),
                                   axis_z= widgets.FloatSlider(description='z', min=-1, max=1, step=0.1, value=axis[2])
                                  )
        position_widgets = dict(xpos = widgets.FloatSlider(description='x [mm]', min=pos[0]-100, max=pos[0]+100, step=0.1, value=pos[0]),
                                ypos = widgets.FloatSlider(description='y [mm]', min=pos[1]-100, max=pos[1]+100, step=0.1, value=pos[1]),
                                zpos = widgets.FloatSlider(description='z [mm]', min=pos[2]-100, max=pos[2]+100, step=0.1, value=pos[2])
                               )

    updateBplot_checkbox = widgets.Checkbox(value=True, description='update B plot', style=dict(description_width='0px'), layout=dict(width='auto'))
    
    sensorsize_widget = widgets.FloatLogSlider(base=2, description='sensor size [mm]', min=-4, max=8, step=1, value=sensorsize)

    all_widgets_sensor = dict(**orientation_widgets, **position_widgets, sensorsize=sensorsize_widget, 
                              update3d=update3d_checkbox, updateBplot=updateBplot_checkbox)
        
    orient_buttons={}
    @debug_view.capture(clear_output=True, wait=True)
    def _on_orient_button_click(b):
        vals = {'angle':0, 'axis_x':0, 'axis_y':0, 'axis_z':0}
        if b.description == 'x':
            vals['axis_y'] = 1 ; vals['angle']=90
        elif b.description == 'y':
            vals['axis_x'] = 1 ; vals['angle']=90
        elif b.description == 'z':
            vals['axis_z'] = 1
        observe_sensor(sensor_id, val=False)
        for k,v in orientation_widgets.items():
            v.value = vals[k]
        observe_sensor(sensor_id, val=True)
            
    for o in ['x','y','z']:
        ob = widgets.Button(description=o, icon='check', tooltip=f'orient along {o}-axis', layout=dict(width='120px'))
        ob.on_click(_on_orient_button_click)
        orient_buttons[f'axis_{o}'] = ob
    orient_button_HBox = widgets.HBox(list(orient_buttons.values()), layout=dict(justify_content='space-between'))
    
    fig_Brecord = go.FigureWidget(layout_template = default_theme)
    fig_Brecord.update_layout(height=150, font_size=8, margin=dict(b=5,t=20,l=5,r=5), yaxis_title='B [mT]')
    for k in ['x','y','z']:
        fig_Brecord.add_scatter(y=(), mode='lines',name='B'+k)

    fig_Brecord_container = widgets.VBox([], layout=dict(max_width='99%'))
    
    
    fig_B_circle = go.FigureWidget(fig_Brecord, layout_template = default_theme)
    
    
    @debug_view.capture(clear_output=True, wait=True)
    def on_recordB_button_click(b):
        if b.description=='stop recording':
            clearBplot()
            fig_Brecord_container.children= []
            b.description='record B'
        elif b.description=='record B':
            w = widgets.HBox([updateBplot_checkbox, clearB_last_button, clearB_button], layout=dict(justify_content='space-between'))
            fig_Brecord_container.children= [fig_Brecord, w]
            b.description = 'stop recording'
                
    @debug_view.capture(clear_output=True, wait=True)
    def clearBplot(last_only=False):
        with fig_Brecord.batch_update():
            for t in fig_Brecord.data:
                if last_only:
                    t.y = t.y[:-1]
                else:
                    t.y=()
            fig_Brecord.layout.title.text = ''
            fig_Brecord.layout.xaxis.title = ''
        bt = Brecord_trace3d            
        if last_only:
            bt.x = bt.x[:-1]
            bt.y = bt.y[:-1]
            bt.z = bt.z[:-1]
        else:
            bt.x = bt.y = bt.z = []
    
    recordB_button= widgets.Button(description='record B', 
                               tooltip='allows recording of B field values seen by the sensor when moving or orienting it \n 3d trace is also shown',
                                   layout=dict(width='auto'), style=dict(button_color='#27AE60'))
    recordB_button.on_click(on_recordB_button_click)
    clearB_button= widgets.Button(description='clear B plot', layout=dict(width='auto'), style=dict(button_color='#E74C3C'))
    clearB_button.on_click(lambda b: clearBplot())
    clearB_last_button= widgets.Button(description='clear last B point', layout=dict(width='auto'), style=dict(button_color='#F5B041'))
    clearB_last_button.on_click(lambda b: clearBplot(last_only=True))
    
    up_widgs = widgets.HBox([continuous_update_checkbox, update3d_checkbox, recordB_button], 
                            layout=dict(justify_content='space-between'))
    
    
    cw, rotation_widgets_ui = define_rotation_widgets(sensor_id)
    
    
    tabs = widgets.Tab([widgets.VBox(list(position_widgets.values()) + [sensorsize_widget, up_widgs, fig_Brecord_container]),
                        widgets.VBox([orient_button_HBox] + list(orientation_widgets.values()) + [up_widgs, fig_Brecord_container]),
                        widgets.VBox(list(rotation_widgets_ui.children) + [fig_B_circle])
                       ])
    
    tabs.set_title(0,'position')
    tabs.set_title(1,'orientation')
    tabs.set_title(2,'rotation array')
    
    for w in all_widgets_sensor.values():
        w.style.description_width='auto'
        w.layout.width='auto'
        w.continuous_update = False
       
    def rename_sensor(b):
        if b.icon=='check':
            sensor_name_container.children =[rename_sensor_button, sensor_opacity_slider, delete_sensor_button] 
            update_sensors_reading(sensor_id)
        else:
            sensor_name_container.children =[widgets.HBox([sensor_name_widget, ok_button], style=dict(description_width='auto'))]
    
    rename_sensor_button = widgets.Button(description='rename', icon='text', button_style='warning',layout=dict(width='auto'))
    rename_sensor_button.on_click(rename_sensor)
    ok_button = widgets.Button(icon='check', button_style='success',layout=dict(width='auto'))
    ok_button.on_click(rename_sensor)
    sensor_opacity_slider = widgets.FloatSlider(description='opacity', value=1, min=0, max=1, style=dict(description_width='auto'))
    widgets.jsdlink((opacity_slider, 'value'), (sensor_opacity_slider, 'value'))
    def update_sensor_opacity(opacity):
        sensor_trace.opacity = opacity
    widgets.interactive(update_sensor_opacity, opacity=sensor_opacity_slider)
    
    sensor_name_container = widgets.HBox([rename_sensor_button, sensor_opacity_slider, delete_sensor_button], 
                                         layout=dict(justify_content='space-between'))
    
    figmag.add_scatter3d(marker_size=2)
    circle_array_trace = figmag.data[-1] 
    
    figmag.add_scatter3d(x=(), y=(), z=(), marker_size=2, name=f'{name} path')
    Brecord_trace3d = figmag.data[-1]
    sensors[sensor_id].update(**{'id': sensor_id, 
                          'sensor_type':sensor_type, 
                          'sensorsize':sensorsize_widget, 
                          'name_widget':sensor_name_widget, 
                          'trace': sensor_trace,
                          'circle_array_trace': circle_array_trace,
                          'Brecord_trace3d': Brecord_trace3d,
                          'traces': [sensor_trace, circle_array_trace, Brecord_trace3d],
                          'properties':sensor_props, 
                          'magpy_sensor': magpy_sensor,
                          'rotation_widgets_ui': rotation_widgets_ui,
                          'widgets': all_widgets_sensor,
                          'fig_Brecord': fig_Brecord,
                          'fig_B_circle' : fig_B_circle,
                          'updateBplot_checkbox': updateBplot_checkbox,
                          'record_B_button': recordB_button,
                          'tabs': tabs,
                          'circle_widgets': cw,
                          'widget': widgets.VBox([sensor_name_container,
                                                  tabs
                                                 ])
                         })
    
    sw = sensors[sensor_id]['widgets']
    for k in ['x','y','z']:
        widgets.jslink((sw[k+'pos'],'value'),(cw[k+'pos'],'value'))
    
    @debug_view.capture(clear_output=True, wait=True)
    def on_selected_tab_change(change):
        if change.new == 2:
            cw['update_circle'].value=True
            for s in sources.values():
                s['tabs'].selected_index = 0
        else:
            cw['update_circle'].value=False

    tabs.observe(on_selected_tab_change, names='selected_index')
    
    sensors_list_accordion.children+=(sensors[sensor_id]['widget'],)
    sensors_list_accordion.selected_index = len(sensors_list_accordion.children)-1
    
    update_sensors_reading(sensor_id)
    
    for w in all_widgets_sensor.values():
        w.id = sensor_id
    observe_sensor(sensor_id, val=True)
    
    if len(sensors)>1:
        delete_all_sensors_button_container.children = [delete_all_sensors_button]
    
    

# %% [markdown]
# # Delete objects

# %% [markdown]
# ## Delete source

# %%
@debug_view.capture(clear_output=True, wait=True)
def on_delete_source_button_click(b):
    delete_source(b.id)
    
@debug_view.capture(clear_output=True, wait=True)
def delete_source(source_id):
    global sources
    figmag.data = [t for t in figmag.data if t not in sources[source_id]['traces']]
    src_col.removeSource(sources[source_id]['magpy_source'])
    sources.pop(source_id, sources)
    sources_list_accordion.children = [v['widget'] for k,v in sources.items()]
    if len(sources)==0:
        for v in streamlines_options.values():
            v['checkbox'].value= False
        graphics_container.children = [graphics_window]
        delete_all_sources_button_container.children = []
    _clear_isosurface_data()
    update_streamlines()
    update_sensors_reading()
    update_sources_titles()

@debug_view.capture(clear_output=True, wait=True)
def delete_all_sources():
    global sources
    _clear_isosurface_data()
    sources_traces = [t for sid in sources.values() for t in sid['traces']]
    figmag.data = [t for t in figmag.data if t not in sources_traces]
    sources.clear()
    for m in sources.values():
        src_col.removeSource(m['magpy_source'])
    sources_list_accordion.children=[]
    for v in streamlines_options.values():
        v['checkbox'].value= False
    graphics_container.children = [graphics_window]
    delete_all_sources_button_container.children = []
    update_sensors_reading()
    update_sources_titles()


# %% [markdown]
# ## Delete sensor

# %%
@debug_view.capture(clear_output=True, wait=True)
def on_delete_sensor_button_click(b):
    delete_sensor(b.id)
        
@debug_view.capture(clear_output=True, wait=True)
def delete_all_sensors():
    global sensors
    sensors_traces = [t for sid in sensors.values() for t in sid['traces']]
    figmag.data = [t for t in figmag.data if t not in sensors_traces]
    sensors_list_accordion.children = []
    update_sensors_reading()
    delete_all_sensors_button_container.children = []
    sensors.clear()
    
@debug_view.capture(clear_output=True, wait=True)
def delete_sensor(sensor_id):
    global sensors
    figmag.data = [t for t in figmag.data if t not in sensors[sensor_id]['traces']]
    sensors.pop(sensor_id, sensors)
    sensors_list_accordion.children = [v['widget'] for k,v in sensors.items()]
    update_sensors_reading()
    if len(sensors)==0:
        delete_all_sensors_button_container.children = []


# %% [markdown]
# ## Delete general

# %%
@debug_view.capture(clear_output=True, wait=True)
def clear_objects_space():
    delete_all_sensors()
    delete_all_sources()
    
@debug_view.capture(clear_output=True, wait=True)
def delete_object(*ids, target_group = None):
    if not ids:
        if target_group == 'sensors':
            delete_all_sensors()
        elif target_group == 'sources':
            delete_all_sources()
        elif target_group == 'all':
            clear_objects_space()
    elif ids and target_group is None:
        for i in ids:
            if i in sensors:
                delete_sensor(id)
            elif i in sources:
                delete_source(id)
            
        
@debug_view.capture(clear_output=True, wait=True)
def _clear_isosurface_data():
    figmag.data[4].x = []
    figmag.data[4].y = []
    figmag.data[4].z = []
    figmag.data[4].value = []


# %% [markdown]
# #  Json handling 

# %% [markdown]
# ## Json functions

# %%
@debug_view.capture(clear_output=True, wait=True)
def get_dict():
    so = {}
    for k,v in sources.items():
        so[k]={'name':v['name_widget'].value,
              'id': v['id'],
              'source_type': v['source_type']}
        props = {}
        for k1,v1 in v['properties'].items():
            if isinstance(v, np.ndarray):
                props[k1] = v1.tolist()
            else:
                props[k1] = v1
        so[k]['properties'] = props
        
    
    se = {}
    for k,v in sensors.items():
        se[k]={'name':v['name_widget'].value,
              'id': v['id'],
              'sensor_type': v['sensor_type'],
              'sensorsize': v['sensorsize'].value}
        props = {}
        for k1,v1 in v['properties'].items():
            if isinstance(v, np.ndarray):
                props[k1] = v1.tolist()
            else:
                props[k1] = v1
        se[k]['properties'] = props
        
    output_dict = {'sources': so, 'sensors':se, 'layout': {}}
    
    return output_dict

@debug_view.capture(clear_output=True, wait=True)
def load_dict(input_dict):
    with figmag.batch_update():
        clear_objects_space()
        for source in input_dict['sources'].values():
            add_source(source_type = source['source_type'], **source['properties'], name=source['name'])
        for sensor in input_dict['sensors'].values():
            sensorsize = 10 if 'sensorsize' not in sensor else sensor['sensorsize']
            add_sensor(sensor_type = sensor['sensor_type'], **sensor['properties'], name=sensor['name'], sensorsize=sensorsize)
        figmag.update_layout(**input_dict['layout'])

@debug_view.capture(clear_output=True, wait=True)
def on_json_file_selector_change(filepath=None):
    if filepath.strip() != '':
        filepath = os.path.join(json_folder,f'{filepath.strip()}.json')
        if os.path.isfile(filepath):
            delete_state_button.disabled = False
            save_state_button.disabled = True
            load_json_button.disabled = False
        else:
            delete_state_button.disabled = True
            save_state_button.disabled = False
            load_json_button.disabled = True
    else:
        delete_state_button.disabled = True
        save_state_button.disabled = False
        load_json_button.disabled = True


@debug_view.capture(clear_output=True, wait=True)
def load_json(filepath):
    with open(filepath, 'r') as fp:
        p_input = json.load(fp)
    load_dict(input_dict = p_input)
            
@debug_view.capture(clear_output=True, wait=True)
def save_json(mydict, filename='input_data.json', folder='saved_sources_sets'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(os.path.join(folder,filename), 'w') as fp:
        json.dump(mydict, fp, sort_keys=True, indent=4)


@debug_view.capture(clear_output=True, wait=True)
def _on_save_state_button_click(b=None):
    fns =  json_file_selector.value
    mydict=  get_dict()
    for i in range(1,100):
        fn = f'{fns}.json'
        if not os.path.isfile(os.path.join(json_folder,fn)):
            break
    save_json(mydict, fn, json_folder)
    update_json_selector_options()
    delete_state_button.disabled = False
    save_state_button.disabled = True

@debug_view.capture(clear_output=True, wait=True)
def update_json_selector_options(reset=False):
    json_paths = [file for file in glob.glob(os.path.join(json_folder, '*.json'))]
    json_filenames = [os.path.basename(f) for f in  json_paths]
    json_file_selector.options = [os.path.splitext(fn)[0] for fn in  json_filenames]
    if reset:
        json_file_selector.value = ''
        
@debug_view.capture(clear_output=True, wait=True)
def _on_reset_state_button_click(b):
    json_file_selector.index = 0
    json_file_selector.value = ''
    update_json_selector_options(reset=True)

@debug_view.capture(clear_output=True, wait=True)
def _on_delete_button_click(b):
    delete_state_button.disabled = True
    save_state_button.disabled = False
    filepath =   os.path.join(json_folder, f'{json_file_selector.value.strip()}.json')
    os.remove(filepath)
    update_json_selector_options(reset=False)


# %% [markdown]
# ## Json widget definitions

# %%
json_folder = 'saved_sources_sets'

save_state_button = widgets.Button(tooltip='save current state to json file', button_style='info', icon = 'save', disabled=True, 
                                        layout=dict(width='auto'))
save_state_button.on_click( _on_save_state_button_click)
reset_state_button = widgets.Button(tooltip='reset and resfresh textbox options', button_style='warning', icon = 'repeat', 
                                         layout=dict(width='auto'))
reset_state_button.on_click( _on_reset_state_button_click)
delete_state_button = widgets.Button(tooltip='delete current state json file', disabled=True, button_style='danger', icon = 'trash', 
                                          layout=dict(width='auto'))
delete_state_button.on_click( _on_delete_button_click)

load_json_button = widgets.Button(tooltip='load state', button_style='success', icon = 'upload', disabled=True, 
                                        layout=dict(width='auto'))
load_json_button.on_click(lambda b: load_json(os.path.join(json_folder,f'{json_file_selector.value.strip()}.json')))
json_file_selector = widgets.Combobox(placeholder='enter name or choose from list')
json_file_selector.style.description_width = '0px'
json_file_selector.layout.flex = '1'
update_json_selector_options(reset=True)
widgets.interactive(on_json_file_selector_change, filepath = json_file_selector)

json_widgets = widgets.HBox([])

json_widgets.children = [json_file_selector, load_json_button,  reset_state_button,  save_state_button,  delete_state_button]

# %% [markdown]
# # UI definitions

# %%
src_col = magpy.Collection()

figmag = go.FigureWidget(layout_template = default_theme)
figmag.add_scatter3d(name='ref coord-sys', mode='lines+text', 
                     x=[0,10,0,0,0,0], y=[0,0,0,10,0,0], z=[0,0,0,0,0,10], 
                     text=['','x','','y','','z'], textfont_color = 'blue')
for k in ['xy', 'xz', 'yz']:
    figmag.add_trace(dict(type='scatter3d', name = f'streamline {k}'))
figmag.add_isosurface()
scene_range = 100
figmag.layout.scene.aspectmode = 'auto'
figmag.layout.scene = dict(xaxis_title = 'x[mm]', yaxis_title = 'y[mm]', zaxis_title = 'z[mm]')
figmag.layout.margin=dict(l=0,r=0,t=10,b=10)
figmag.layout.showlegend = True
figmag.layout.legend.orientation = 'h'


sources={}
sensors={}
add_sensor_button = widgets.Button(description='3d-Hall-sensor', icon='plus', layout=dict(flex='1'), style=dict(button_color='#138D75'))
add_sensor_button.on_click(lambda b: add_sensor(sensor_type='hall-3d'))
delete_all_sensors_button = widgets.Button(description='delete all sensors', icon='trash', layout=dict(flex='1'), button_style='danger')
@debug_view.capture(clear_output=True, wait=True)
def confirm_all_sensors_deletion(b):
    confirm_button = widgets.Button(description='ok', icon='check', layout=dict(width='auto'), button_style='success')
    confirm_button.on_click(lambda b: delete_object(target_group = 'sensors'))
    def on_cancel_click(b):
        if len(sensors)>1:
            delete_all_sensors_button_container.children = [delete_all_sensors_button]
        else:
            delete_all_sensors_button_container.children = []
    cancel_button = widgets.Button(description='cancel', icon='times', layout=dict(width='auto'), button_style='warning')
    cancel_button.on_click(on_cancel_click)
    delete_all_sensors_button_container.children = [widgets.HBox([widgets.HTML('Are you sure ? (cannot be undone)'), confirm_button, cancel_button])]

delete_all_sensors_button.on_click(confirm_all_sensors_deletion)

delete_all_sensors_button_container = widgets.HBox()
add_sensor_buttons = widgets.VBox([widgets.HBox([add_sensor_button])])

object_index=0
@debug_view.capture(clear_output=True, wait=True)
def on_add_object_button_click(b):
    global object_index
    i = 50*object_index
    if b.description=='box':
        add_source(source_type='box', pos=(i,i,i))
    if b.description=='cylinder':
        add_source(source_type='cylinder', dim=(60,60), pos=(i,i,i))
    if b.description=='sphere':
        add_source(source_type='sphere', dim=70, pos=(i,i,i))
    if b.description=='dipole':
        add_source(source_type='dipole', moment=(0,0,10), pos=(i,i,i))
    if b.description=='line current':
        add_source(source_type='lineCurrent', vertices=[(-100,0,0),(100,0,0)], pos=(i,i,i))
    if b.description=='circular current':
        add_source(source_type='circularCurrent', dim=70, pos=(i,i,i))
        
    #object_index+=1
        



add_box_button = widgets.Button(description='box', icon='plus', layout=dict(flex='1'), style=dict(button_color='#EC7063'))
add_box_button.on_click(on_add_object_button_click)
add_cylinder_button = widgets.Button(description='cylinder', icon='plus', layout=dict(flex='1'), style=dict(button_color='#8E44AD'))
add_cylinder_button.on_click(on_add_object_button_click)
add_sphere_button = widgets.Button(description='sphere', icon='plus', layout=dict(flex='1'), style=dict(button_color='#3498DB'))
add_sphere_button.on_click(on_add_object_button_click)
add_dipole_button = widgets.Button(description='dipole', icon='plus', layout=dict(flex='1'), style=dict(button_color='#2ECC71'))
add_dipole_button.on_click(on_add_object_button_click)
add_line_button = widgets.Button(description='line current', icon='plus', layout=dict(flex='1'), style=dict(button_color='#D4AC0D'))
add_line_button.on_click(on_add_object_button_click)
add_circle_button = widgets.Button(description='circular current', icon='plus', layout=dict(flex='1'), style=dict(button_color='#E67E22'))
add_circle_button.on_click(on_add_object_button_click)
delete_all_sources_button = widgets.Button(description='delete all sources', icon='trash', layout=dict(flex='1'), button_style='danger')
delete_all_sources_button.target = 'sources'
@debug_view.capture(clear_output=True, wait=True)
def confirm_all_sources_deletion(b):
    confirm_button = widgets.Button(description='ok', icon='check', layout=dict(width='auto'), button_style='success')
    confirm_button.on_click(lambda b: delete_object(target_group = 'sources'))
    def on_cancel_click(b):
        if len(sensors)>1:
            delete_all_sources_button_container.children = [delete_all_sources_button]
        else:
            delete_all_sources_button_container.children = []
    cancel_button = widgets.Button(description='cancel', icon='times', layout=dict(width='auto'), button_style='warning')
    cancel_button.on_click(on_cancel_click)
    delete_all_sources_button_container.children = [widgets.HBox([widgets.HTML('Are you sure ? (cannot be undone)'), confirm_button, cancel_button])]
delete_all_sources_button.on_click(confirm_all_sources_deletion)
delete_all_sources_button_container = widgets.HBox()
add_source_buttons = widgets.VBox([widgets.HBox([add_box_button, add_cylinder_button, add_sphere_button]),
                                   widgets.HBox([add_dipole_button, add_line_button, add_circle_button])
                                  ])


continuous_update_checkbox = widgets.Checkbox(value=False, description = 'continuous slider update', layout=dict(width='auto'), style=dict(description_width='auto'))
continuous_update_checkbox.observe(on_continuous_update_change, names='value')
        
streamlines_options, streamline_widgets = _define_streamlines_widgets()
streamlines_accordion = widgets.Accordion([streamline_widgets])
streamlines_accordion.set_title(0,'Streamlines')
streamlines_accordion.selected_index = None

isosurface_options, isosurface_widgets = _define_isosurface_widgets()
isosurface_accordion = widgets.Accordion([isosurface_widgets])
isosurface_accordion.set_title(0,'Isosurface')
isosurface_accordion.selected_index = None


def set_opacity(opacity):
    for m in sources.values():
        m['trace'].opacity = opacity
opacity_slider =  widgets.FloatSlider(description='objects opacity', min=0, max=1, step=0.1, value=1, style=dict(description_width='auto'))
widgets.interactive(set_opacity, opacity =opacity_slider)

def on_figmag_height_change(h):
    figmag.layout.height = h
figmag_height_slider = widgets.FloatSlider(description='fig3d height', min=300, max=1200, value=600)
widgets.interactive(on_figmag_height_change, h=figmag_height_slider)

def set_fig_template(template):
    figmag.layout.template = template
    for s in sensors.values():
        s['fig_Brecord'].layout.template = template
        s['fig_B_circle'].layout.template = template
fig_template_dropdown = widgets.Dropdown(description='theme', value = default_theme, 
                                         options=['plotly_grey', 'plotly', 'plotly_white'], 
                                         style=dict(description_width='auto'), layout=dict(width='150px'))
widgets.interactive(set_fig_template, template=fig_template_dropdown)


auto_scene_range_checkbox = widgets.ToggleButton(value=False, icon='sliders', description='auto', layout=dict(width='auto'), style=dict(description_width='0px'))

scene_range_textbox = widgets.BoundedFloatText(description='range', value=100, min=0, max=1000, layout=dict(width='150px'), style=dict(description_width='60px'))
scene_range_widgets = widgets.HBox(layout=dict(flex_flow='wrap'))

@debug_view.capture(clear_output=True, wait=True)
def update_scene_range(auto=False, sr=100):
    sc = figmag.layout.scene
    if not auto:
        auto_scene_range_checkbox.description = 'auto'
        scene_range_widgets.children = [scene_range_textbox]
        with figmag.batch_animate():
            sc.xaxis.autorange = False
            sc.yaxis.autorange = False
            sc.zaxis.autorange = False
            sc.xaxis.range = [-sr,sr]
            sc.yaxis.range = [-sr,sr]
            sc.zaxis.range = [-sr,sr]
            figmag.layout.scene.aspectmode = 'cube'
    else:
        auto_scene_range_checkbox.description = 'scene range'
        scene_range_widgets.children = []
        with figmag.batch_animate():
            sc.xaxis.autorange = True
            sc.yaxis.autorange = True
            sc.zaxis.autorange = True
            figmag.layout.scene.aspectmode = 'auto'
    
    sr = scene_range_textbox.value
    
    for k0,v in sources.items():
        w = v['widgets']
        for k in ['xpos', 'ypos', 'zpos', 'Lx', 'Ly', 'Lz']:
            observe_source(k0, val=False)
            if k in w:
                w[k].min = -sr if not k.startswith('L') else 0
                w[k].max = sr if not k.startswith('L') else 2*sr
            observe_source(k0, val=True)
    
    for k0,v in sensors.items():
        w = v['widgets']
        observe_sensor(k0, val=False)
        for k in ['xpos', 'ypos', 'zpos']:
            w[k].min = -sr
            w[k].max = sr
        observe_sensor(k0, val=True)
    
widgets.interactive(update_scene_range, auto = auto_scene_range_checkbox, sr = scene_range_textbox)
scene_range_container = widgets.HBox([scene_range_widgets, auto_scene_range_checkbox])

update_scene_range()

@debug_view.capture(clear_output=True, wait=True)
def f(obj,x,y,z):
    for k,r in zip(['x','y','z'], [x,y,z]):
        sr = isosurface_options['sr'+k]
        sr.min, sr.max = r
        sr.value = r
figmag.layout.scene.on_change(f, ('xaxis', 'range'), ('yaxis', 'range'), ('zaxis', 'range'))

data_loader_window = widgets.Accordion([json_widgets])
data_loader_window.set_title(0,'Json Data Loader')
data_loader_window.selected_index=None
data_loader_window.layout.min_width='500px'

sensors_list_accordion = widgets.Accordion([])
sensors_window = widgets.Accordion([widgets.VBox([add_sensor_buttons, sensors_list_accordion, delete_all_sensors_button_container])])
sensors_window.set_title(0,'Sensors')
sensors_window.layout.min_width='500px'

sources_list_accordion = widgets.Accordion([])
sources_window = widgets.Accordion([widgets.VBox([add_source_buttons, sources_list_accordion, delete_all_sources_button_container])])
sources_window.set_title(0,'Sources')
sources_window.layout.min_width='500px'
update3d_all_updates_checkbox = widgets.Checkbox(value=True, description='update all 3d objects', style=dict(description_width='20px'), layout=dict(width='auto'))
graphics_window = widgets.Accordion([widgets.VBox([figmag, 
                                                      widgets.HBox([figmag_height_slider,
                                                                    opacity_slider, 
                                                                    fig_template_dropdown,
                                                                    scene_range_container,
                                                                    update3d_all_updates_checkbox],
                                                                  layout=dict(flex_flow='wrap', justify_content='space-around'))],
                                                     layout=dict(max_width='95%'))])
graphics_window.set_title(0, 'Graphics')
graphics_container = widgets.VBox([graphics_window])
graphics_container.layout.flex='600px'

debug_accordion = widgets.Accordion([debug_view])
debug_accordion.set_title(0,'Exceptions info')
debug_accordion.selected_index=None

objects_window = widgets.VBox([data_loader_window, sensors_window, sources_window])
objects_window.layout.flex='510px'
app = widgets.VBox([widgets.HBox([objects_window,
                                 graphics_container],
                                      layout=dict(border='solid 1px', flex_flow='wrap')),
                    debug_accordion
                   ])


# %% [markdown]
# # Testing

# %%
app

# %% [raw]
# @debug_view.capture(clear_output=True, wait=True)
# def update_object_from_rotation(object_id, angle=0 , axis=(0,0,1), anchor=None):
#     observe_sensor(object_id, val=False)
#     if object_id in sources:
#         sp = sources[object_id]
#         mp = sp['magpy_source']
#     else:
#         sp = sensors[object_id]
#         mp = sp['magpy_sensor']
#     if anchor is None:
#         anchor = mp.position
#     mp.rotate(angle=angle, axis=axis, anchor=anchor)
#     w = sp['widgets']
#     w['xpos'].value, w['ypos'].value, w['zpos'].value = mp.position
#     w['angle'].value = mp.angle
#     w['axis_x'].value, w['axis_y'].value, w['axis_z'].value = mp.axis
#     observe_sensor(object_id, val=True)
# for a in range(100):
#     update_object_from_rotation('hall-3d_01', angle=3, anchor=(0,0,0))

# %% [raw]
# sensor_idt = np.linspace(0,360, 100)
# r = 10
# x = r*np.cos(np.deg2rad(t))
# y = r*np.sin(np.deg2rad(t))
# z = np.zeros(len(t))
# pts = np.array([x,y,z]).T
# s = sensors['hall-3d_01']
# sw = s['widgets']
# s['updateBplot_checkbox'].value = False
# update3d_all_updates_checkbox.value = False
# magB=[]
# ts = figmag.data[-1]
# for p in pts:
#     observe_sensor('hall-3d_01', val=False)
#     sw['xpos'].value, sw['ypos'].value, sw['zpos'].value = p
#     observe_sensor('hall-3d_01', val=True)
#     #magB.append(s['magpy_sensor'].getB(*so))
# #magB

# %% [raw]
# import ipywidgets as widgets
# from traitlets import validate
#
# class UnboudedFloatSlider(widgets.FloatSlider):
#     '''Subclassing Floatslider in order to accept values out of bounds'''
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         
#     @validate('value')
#     def _validate_value(self, proposal):
#         value = proposal['value']
#         if value > self.max:
#             self.max = 2*value - self.min
#         elif value < self.min:
#             print(2*value - self.max)
#             self.min = 2*value - self.max
#         return value
