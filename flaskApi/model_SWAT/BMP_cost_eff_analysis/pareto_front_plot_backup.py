# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:47:19 2020

@author: Shaobin
"""

def pareto_front_plot(name, sw):
    # scores1 = scores('phosphorus', 33)
    scores1 = scores(name, sw)
    pareto = identify_pareto(scores1)
    pareto = np.insert(pareto,[0], 0)
    pareto_front = scores1[pareto]
    pareto_front[:,1] = pareto_front[:,1]*-1
    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values
    
    BMP_list = ['BMP' + str(i) for i in range(1,55)]
    x = scores1[:, 0]
    y = scores1[:, 1]*-1
    z = y/x
    if name == 'phosphorus':
        z[24] = 2000
    
    f, ax = plt.subplots(figsize=(10,5))
    cmap = cm.get_cmap('seismic')
    points = ax.scatter(x, y, c=z, s=50, edgecolor='k', cmap=cmap)
    ax.plot(pareto_front[:,0], pareto_front[:,1], color='r')
    for i, txt in enumerate(BMP_list):
        ax.annotate(txt, (x[i], y[i]))
    ax.set_xlabel(name + ' reduction (kg/ha)', fontsize= 12)
    ax.set_ylabel('Crop net revenue loss ($/ha)', fontsize=12)   
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                 loc="center right", title="Cost ($/kg removal)")
    f.colorbar(points, label = 'Cost ($/kg removal)')
    plt.show()
    return

# pareto_front_plot('nitrate', 33)