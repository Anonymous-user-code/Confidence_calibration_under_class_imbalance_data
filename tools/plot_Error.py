'''
The code of plotting Fig. 3
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

step = 0.01
p1 = np.arange(0,1+step,step)
beta = [1/100,1/10,10,100]
p0 = 1-p1
colors = ["skyblue", "lightgreen", "peachpuff", "wheat"]
colors = ["lightcoral", "palevioletred", "plum", "skyblue"]
colors = ["cornflowerblue", "lightseagreen", "sandybrown", "orchid"]

for i in range(len(beta)):
    E = abs(1-1/(p1+beta[i]*p0))
    plt.plot(p1,E,linewidth = 5,color=colors[i])
plt.xlabel("${P({Y_k} = 1|X = x)}$",fontsize=35,fontname="Times New Roman")
plt.ylabel("Relative error $(\\epsilon$)",fontsize=35,fontname="Times New Roman")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tick_params(axis='both', labelsize=35)
plt.legend(["$\\beta_{1,0}$=0.01 (Head class)","$\\beta_{1,0}$=0.1 (Head class)","$\\beta_{1,0}$=10 (Tail class)","$\\beta_{1,0}$=100 (Tail class)"],prop={"family": "Times New Roman","size":35})
plt.ylim(0,2)
plt.xlim(0,1)
plt.show()