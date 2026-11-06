import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_scene(scene, paths=None, title="RoboBallet AI — coordinated workcell", save=None):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-.5, scene.width-.5); ax.set_ylim(-.5, scene.height-.5)
    ax.set_xticks(range(scene.width)); ax.set_yticks(range(scene.height))
    ax.grid(True, alpha=.2); ax.set_aspect("equal")
    for x,y in scene.obstacles:
        ax.add_patch(Rectangle((x-.5,y-.5),1,1,color="#303846"))
    for i,(x,y) in enumerate(scene.starts):
        ax.scatter(x,y,s=150,marker="s",edgecolor="black",zorder=5)
        ax.text(x,y,f" R{i}",ha="left",va="bottom",weight="bold")
    for i,(x,y) in enumerate(scene.tasks):
        ax.scatter(x,y,s=110,marker="*",edgecolor="black",zorder=5)
        ax.text(x,y,f" T{i}",ha="left",va="bottom",fontsize=8)
    if paths:
        for i,path in enumerate(paths):
            if path:
                ax.plot([p[0] for p in path],[p[1] for p in path],linewidth=2,alpha=.8,label=f"Robot {i}")
    ax.set_title(title); ax.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    fig.tight_layout()
    if save: fig.savefig(save,dpi=160,bbox_inches="tight")
    return fig
