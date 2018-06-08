from visdom import Visdom
import numpy as np


viz = Visdom()
viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)