import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# %%
tail_location = np.zeros((3, 3))
tip_location = np.asarray([[1, 2, 3], [1.5, 2, 2.5], [2, -2, 1]])

x, y, z = tail_location[:,0], tail_location[:,1], tail_location[:,2]
u, v, w = tip_location[:,0], tip_location[:,1], tip_location[:,2]
# %
# fig, ax = plt.subplots(figsize=(8, 6))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(x,y,z,u,v,w)
Axes3D.quiver(x, y, z, u, v, w)
plt.show()

x, y, z = np.meshgrid(np.arange(0.8, 1, 0.2),
                      np.arange(0.8, 1, 0.2),
                      np.arange(0.8, 1, 0.8))

# %%
from Mediumrare import db_tools
import pandas as pd
conn = db_tools.get_conn()
query = 'SELECT title, blog_url from mediumcleanfull ORDER BY id'
titles = conn.execute(query).fetchall()

df = pd.read_sql(query, conn)
df['channel'] = df.blog_url.map(lambda x: x.split('/')[3])
df.head()
