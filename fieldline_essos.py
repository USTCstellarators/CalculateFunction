import jax.numpy as jnp
from scipy.optimize import minimize
from essos.coils import Coils_from_simsopt
from essos.dynamics import Tracing
from mymisc import rzp2curverznfp
import numpy as np
import os
from time import time
import jax.numpy as jnp
from jax import block_until_ready
import matplotlib.pyplot as plt
from essos.fields import BiotSavart
from essos.dynamics import Tracing
from fieldline import findax


def run_essos_tracing(
    coils,
    tmax= 1000,
    nfieldlines_per_core= 3,
    number_of_processors_to_use=3,
    ma=None,
    len=0.1,
    trace_tolerance=1e-8,
    num_steps=6000,
    plot: bool = True,
    save: bool = False,
    outputname: str = "poincareplot.png"
):
    """
    Run ESSOS magnetic field line tracing.

    Parameters:
    - coils: Coils object from ESSOS (e.g., from Coils_from_json)
    - tmax: Maximum tracing time
    - nfieldlines_per_core: Number of field lines per core
    - number_of_processors_to_use: Number of processors (should divide total field lines)
    - R0: 1D jnp array of initial R positions (length = number_of_processors_to_use * nfieldlines_per_core)
    - trace_tolerance: Absolute and relative tolerance for tracer
    - num_steps: Number of time steps
    - plot: If True, plots the 3D trajectory and Poincaré section
    - save_vtk: If True, saves output in .vtk format
    - output_prefix: Prefix for VTK output files

    Returns:
    - tracing: ESSOS Tracing object
    - trajectories: Traced trajectories (3D array)
    """

    # Set number of processors for JAX XLA
    os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'

    # Derived parameters
    nfieldlines = number_of_processors_to_use * nfieldlines_per_core
    if ma==None:
        ma=findax(coils)
    R0 = jnp.linspace(ma[0],ma[0]+len, nfieldlines)
    # Initial positions
    Z0 = jnp.zeros(nfieldlines)
    phi0 = jnp.zeros(nfieldlines)
    initial_xyz = jnp.array([R0 * jnp.cos(phi0), R0 * jnp.sin(phi0), Z0]).T
    coils_essos = Coils_from_simsopt(coils)
    # Run tracing
    time0 = time()
    tracing = block_until_ready(Tracing(
        field=BiotSavart(coils_essos),
        model='FieldLineAdaptative',
        initial_conditions=initial_xyz,
        maxtime=tmax,
        times_to_trace=num_steps,
        atol=trace_tolerance,
        rtol=trace_tolerance
    ))
    print(f"ESSOS tracing took {time() - time0:.2f} seconds")
    trajectories = tracing.trajectories

    # Plot results
    if plot:
        fig = plt.figure(figsize=(9, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        tracing.plot(ax=ax1, show=False)
        tracing.poincare_plot(ax=ax2, show=False, shifts=[0, jnp.pi / 2])
        plt.tight_layout()
        if save:
            plt.savefig(outputname)
        else:plt.show()


    return tracing, trajectories



def findax_essos(coils, rz0, nfp, phi0=0.0, bounds=None, **kwargs):
    """
    用 ESSOS 的 Tracing 寻找闭合磁力线起点。
    搜索 (R, Z) 使得沿场线走 Δφ = 2π/nfp 后回到原点。
    返回 ndarray: [R*, Z*]
    """
    if bounds is None:
        bounds = [(rz0[0] - 0.05, rz0[0] + 0.05), (-0.05, 0.05)]

    coils_essos = Coils_from_simsopt(coils)
    dphi = 2 * jnp.pi / nfp

    # 当 φ 达到 phi0 + dphi 时停止（Event）
    def condition_phi(t, y, args, **kw):
        x, y_, z = y
        phi = jnp.arctan2(y_, x)
        dphi_current = jnp.mod(phi - phi0 + jnp.pi, 2 * jnp.pi) - jnp.pi
        return jnp.squeeze(dphi_current - dphi)  # 必须返回标量

    def fun(rz):
        R, Z = rz
        x0 = R * jnp.cos(phi0)
        y0 = R * jnp.sin(phi0)
        z0 = Z
        init_cond = jnp.array([[x0, y0, z0]])  # [x, y, z]

        tr = Tracing(
            model='FieldLineAdaptative',
            field=coils_essos,
            initial_conditions=init_cond,
            times_to_trace=2,      # 只保存起点/终点
            maxtime=1e+2,          # 兜底；实际由事件提前停止
            timestep=1e-5,
            rtol=1e-8,
            atol=1e-8,
            condition=condition_phi,
            **kwargs
        )

        x1, y1, z1 = tr._trajectories[0, -1, :]
        R1 = jnp.sqrt(x1**2 + y1**2)
        Z1 = z1
        return float((R1 - R)**2 + (Z1 - Z)**2)

    res = minimize(fun, rz0, method='L-BFGS-B', bounds=bounds)
    return res.x  # ndarray [R*, Z*]


def fullax_essos(coils, nfp, rz0=None, phi0=0.0, bounds=None,
                 order=10, nstep=720, **kwargs):

    R0, Z0 = findax_essos(coils, rz0=rz0, nfp=nfp, phi0=phi0, bounds=bounds, **kwargs)
    print(f"[fullax_essos] Found axis point: R={R0:.6f}, Z={Z0:.6f}")

    coils_essos = Coils_from_simsopt(coils)

    x0 = R0 * jnp.cos(phi0)
    y0 = R0 * jnp.sin(phi0)
    z0 = Z0
    init_cond = jnp.array([[x0, y0, z0]])

    dphi_target = 2 * jnp.pi / nfp

    def condition_full_period(t, y, args, **kw):
        xx, yy, zz = y
        phi = jnp.arctan2(yy, xx)
        dphi_cur = jnp.mod(phi - phi0 + jnp.pi, 2 * jnp.pi) - jnp.pi
        return jnp.squeeze(dphi_cur - dphi_target)

    tracing = Tracing(
        model='FieldLineAdaptative',
        field=coils_essos,
        initial_conditions=init_cond,
        times_to_trace=nstep,
        # maxtime=1e+3,        # 兜底；理论上事件会提前停
        # timestep=1e-5,
        rtol=1e-8,
        atol=1e-9,
        condition=condition_full_period,
        **kwargs
    )

    traj = tracing._trajectories[0]  # (nstep, 3) -> [x, y, z]
    print(f"[fullax_essos] Traced {traj.shape[0]} points along the field line.")
    print(traj)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    Z = z
    line = jnp.vstack([R, phi, Z]).T  # (nstep, 3) -> [R, phi, Z]

    # Step 3: 拟合 Fourier 曲线（按你的最小改动版，仅加入 nfp）
    ma = rzp2curverznfp([np.asarray(line)], order=order, nfp=nfp)
    return ma,traj


if __name__ == "__main__":
    from simsopt._core import load, save
    from simsopt.field import coils_to_focus, BiotSavart
    from simsopt.geo import SurfaceRZFourier,CurveXYZFourier,plotting
    import matplotlib.pyplot as plt
    ID = 400049# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000  
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')
    x=surfaces[0].gamma()[0,:,0]
    y=surfaces[0].gamma()[0,:,1]

    r0=jnp.mean(jnp.sqrt(x**2+y**2))
    print(r0)
    # ma=findax_essos(coils,rz0=[0.876,0],nfp=surfaces[-1].nfp)
    ma,traj=fullax_essos(coils,rz0=[r0,0],nfp=surfaces[-1].nfp)
    plotting.plot([ma])

    items=coils.copy()
    items.append(surfaces[0])
    ax=plotting.plot(items,show=False)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    ax.scatter(x, y, z, color='blue', s=100, label='标记点')

    ax.scatter(r0, 0, 0, color='red', s=100, label='标记点')

    plt.show()
    # poincareplot(cpcoils,rz0=[1,0],len=0.1,show=True)#磁轴位置x: [ 8.763e-01  4.552e-06]







