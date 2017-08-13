import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
lads = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lad_height = 24 + 4 * np.random.randn(lads)

plt.hist([grey_height,lad_height],stacked=True,color=['r','b'])
plt.show()