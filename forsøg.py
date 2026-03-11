import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import Markdown as md
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pmdarima as pm
from math import sqrt
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Define the AR terms in the simulated process
arparams = np.array([0.8, -0.3])
ar = np.r_[1, -arparams]

# Ignore the MA terms for now and keep them equal to 0
maparams = np.array([0])
ma = np.r_[1, maparams]

# Create the model to generate the samples
arma_process = sm.tsa.ArmaProcess(ar, ma)
data = arma_process.generate_sample(500)

# Split between a training and a testing dataset
n = int(0.8*len(data))
N = int(len(data))
train, test = pm.model_selection.train_test_split(data, train_size=n)

# Plot the result
plt.figure(figsize=(8, 4), dpi=100)
plt.plot(np.arange(1,n+1), train)
plt.plot(np.arange(n+1,N+1), test)
plt.legend(["Training set", "Testing set"])
plt.tight_layout()
plt.xlabel("t")
plt.ylabel("x")
plt.grid(alpha=0.25)
plt.show()
