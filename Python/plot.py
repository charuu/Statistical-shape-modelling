import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np, pandas as pd ,seaborn as sns
import matplotlib.pyplot as plt

rv = np.random.multivariate_normal([0, 0], [[4, 6], [6, 9]],100)

plt.xlim(-5,10)
plt.ylim(-5,10)
plt.scatter(rv[:,0],rv[:,1])
plt.show()