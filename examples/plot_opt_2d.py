import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from optlang import Model, Variable, Constraint, Objective
import sympy
from functools import reduce

def region_plot(constraints, var1, var2, N1=500, N2=500, ax=None):
    ax = ax or plt.axes()
    model = Model(name='My model')
    model.add(constraints)
    plot_LP(model, var1=var1, var2=var2, N1=N1, N2=N2, ax=ax, plot_objective=False)

def plot_LP(model, var1, var2, N1=500, N2=500, ax=None, plot_objective=True):
    ax = ax or plt.axes()
    x,x1,x2 = var1
    y,y1,y2 = var2
    xx, yy = sympy.symbols('xx, yy')
    exls = []
    for v,vv in ((x,xx),(y,yy)):
        if v.lb is not None:
            exls.append(sympy.lambdify((xx,yy), vv>=v.lb))
        if v.ub is not None:
            exls.append(sympy.lambdify((xx,yy), vv<=v.ub))
    exls.extend(sympy.lambdify((xx,yy), c.expression.subs({x:xx,y:yy}) >= c.lb) 
            for c in model.constraints if c.lb is not None
           )
    exls.extend(sympy.lambdify((xx,yy), c.expression.subs({x:xx,y:yy}) <= c.ub) 
            for c in model.constraints if c.ub is not None
           )
    xs = np.linspace(x1,x2,N1+1)
    ys = np.linspace(y1,y2,N2+1)
    X, Y = np.meshgrid(xs,ys)
    Z = reduce(lambda B1, B2:B1&B2, [exl(X,Y) for exl in exls]).astype(int)
    lb = (.3,.3,1)
    ax.contourf(X,Y,Z, [0.99999, 1.00001], colors=[lb,lb])
    ax.contour(X,Y,Z, [], colors='b')
    if x.type=='integer' or y.type=='integer':
        ps = [(x0,y0) for x0 in range(x1,x2) for y0 in range(y1, y2)
                      if all(exl(x0,y0) for exl in exls)]
        ax.plot([x for x,y in ps], [y for x,y in ps], 'ok')
#    elif any(v.type=='integer' for v in (x,y)):
#        if y.type=='integer':
#            x,y=y,x
    if x.type=='integer' and y.type=='continuous':
        for x0 in range(x1, x2):
            y_x0 = [y0 for y0 in ys if all(exl(x0,y0) for exl in exls)]
            if y_x0:
                ax.plot([x0,x0], [min(y_x0), max(y_x0)],'k')
    if x.type=='continuous' and y.type=='integer':
        for y0 in range(y1, y2):
            x_y0 = [x0 for x0 in xs if all(exl(x0,y0) for exl in exls)]
            if x_y0:
                ax.plot([min(x_y0), max(x_y0)], [y0, y0],'k')

    if plot_objective:
        objl = sympy.lambdify((xx,yy),
                       model.objective.expression.subs({x:xx,y:yy}))
        Z = objl(X,Y)
        ax.contour(X,Y,Z, [model.objective.value if model.status=='optimal' else 0], colors='g')
        arrow_x, arrow_y = ((model.variables['x'].primal, model.variables['y'].primal) 
            if model.status=='optimal' else (0,0)
        )
        arrow_vx = np.float32( model.objective.expression.coeff(x))
        arrow_vy = np.float32(model.objective.expression.coeff(y))
        s = (1 if model.objective.direction=='max' else -1) * (x2-x1) * 0.1 / np.sqrt(arrow_vx**2+arrow_vy**2)
        ax.arrow(arrow_x, arrow_y, s*arrow_vx, s*arrow_vy, head_width=0.05, head_length=0.1, fc='g', ec='g')

        if model.status=='optimal':
            ax.plot([arrow_x], [arrow_y], 'r*')
        ax.set_title(model.status)
