
# coding: utf-8

# In[1]:
import pandas as pd
import ipdb
# In[2]:
# local
#%load_ext autoreload
# get_ipython().magic('aimport data_proc.ers')
from data_proc.ers import GDM, get_gdm_h5path


# ##### gdm instance creation takes time, because database schema is mapped to classes
# ###### TODO: allow input of stored schema

# In[3]:

gdm = GDM()
ipdb.set_trace()

# In[4]:

queries = gdm.get_standard_queries()
q_HDF = queries['HDF']
q_SQL = queries['SQL']


# In[5]:

classes = gdm.classes
session = gdm.session

runs = classes.Run
measurements = classes.Measurement
shifts = classes.Shift
customers = classes.Customer
stored_channels = classes.StoredChannel
channels = classes.Channel
systems = classes.System
channel_types = classes.ChannelType

# data tables
sql_furis_object = gdm.classes.Furis_Object
sql_furis_suspect = gdm.classes.Furis_Suspect


# In[6]:

# basename = '1705'
basename = '1802'
expr_meas = measurements.BaseName.contains(basename)
system = 'Furis'
expr_syst = systems.Name == system

q_hdf_date_sys = q_HDF.filter(expr_meas).filter(expr_syst)
q_sql_date_sys = q_SQL.filter(expr_meas).filter(expr_syst)

run_id = runs.ID
ipdb.set_trace()
for name, q in [('HDF', q_hdf_date_sys),
                ('SQL', q_sql_date_sys)]:
    df = pd.DataFrame(list(q.values(run_id, measurements.BaseName, systems.Name, channels.Name)),
                      columns=['run_ID', 'meas_Basename', 'syst_Name', 'chan_Name'])
    print(f'# {name} - Systems  #\n {df.syst_Name.unique()}\n')
    print(f'# {name} - Channels #\n {df.chan_Name.unique()}\n')


# In[7]:

hdf_channel = 'Echo data'
sql_channel = 'Object'

q_hdf_runID, _ = gdm.filter_runID(q_hdf_date_sys, system, hdf_channel)
q_sql_runID, _ = gdm.filter_runID(q_sql_date_sys, system, sql_channel)

#q_runID = q_hdf_runID
q_runID = q_sql_runID  # is this always created together with hdf_channel?

runs_sel = pd.Series(
    {q.Measurement.BaseName: q.Run.ID for q in q_runID.all()}, name='run_id').to_frame()
runs_sel.to_csv(f'_{basename}-runs-id.csv', sep=';')
runs_sel


# ###### have a look at the full (?) relationships for one run example

# In[8]:

run_id = int(runs_sel.values[0])

pd.read_sql(q_sql_runID.filter(runs.ID == run_id).statement, session.bind)


# In[9]:

print(q_sql_runID.filter(runs.ID == run_id).statement)


# ###### store all Furis objects of given basename pattern

# In[10]:

q_f_o = session.query(sql_furis_object)
qq_f_o = q_f_o.join(runs, sql_furis_object.RunID == runs.ID).filter(
    runs.ID.in_(runs_sel.values.tolist()))
objects = pd.read_sql(qq_f_o.statement, session.bind)

if True:
    objects.to_hdf(f'_{basename}-furis-object.h5', 'objects', format='t')

print(objects.shape)
objects.head()


# ###### local cache of Furis echo data

# In[11]:
ipdb.set_trace()
import os
import shutil
from IPython.display import display_pretty

gdm_data_dir = r'\\NLAMFSRFL01.railinfra.local\GDM_RO$\Live\Data'
cache_dir = os.path.join(r'c:\DATA\GDM', system)
os.makedirs(cache_dir, exist_ok=True)

for run, row in runs_sel.iterrows():
    gdm_id = row.run_id
    fn = get_gdm_h5path(gdm_id, system)
    fnpath = os.path.join(gdm_data_dir, fn)
    fn_new = os.path.basename(fn)
    fnpath_new = os.path.join(cache_dir, fn_new)
    try:
        if os.path.isfile(fnpath_new):
            # TODO: check also file size
            raise ValueError
        shutil.copy(fnpath, cache_dir)
        display_pretty('{} copied.'.format(fnpath))
    except ValueError:
        display_pretty('File {} is already in cache directory.'.format(fn_new))
    except OSError:
        display_pretty('{} NOT copied.'.format(fnpath))
