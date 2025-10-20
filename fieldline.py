import numpy as np
from math import pi
import matplotlib.pyplot as plt
from coilpy import Coil as cl
from coilpy.misc import poincare_plot,print_progress
from scipy.optimize import minimize
from simsopt.geo.curverzfourier import CurveRZFourier
from numpy import arctan
from mpl_toolkits.mplot3d import Axes3D
from simsopt.geo import CurveXYZFourier,plotting
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import os
from simsopt.field import (InterpolatedField, coils_via_symmetries, SurfaceClassifier,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
import time
from simsopt.mhd import Vmec
from mymisc import rzp2curverz,from_simsopt,nml_to_focus

def findrz0(coils):
    x=0.3*max(coils.data[0].x)+0.7*min(coils.data[0].x)
    y=0.3*max(coils.data[0].y)+0.7*min(coils.data[0].y)
    print('rz0=',np.sqrt(x**2+y**2))
    return np.sqrt(x**2+y**2)

def tracing(bfield, r0, z0, phi0=0.0, niter=100, nfp=1, nstep=1, **kwargs):
    """Trace magnetic field line in toroidal geometry

    Args:
        bfield (callable): A callable function.
                          The calling signature is `B = bfield(xyz)`, where `xyz`
                          is the position in cartesian coordinates and `B` is the
                          magnetic field at this point (in cartesian coordinates).
        r0 (list): Initial radial coordinates.
        z0 (list): Initial vertical coordinates.
        phi0 (float, optional): The toroidal angle where the poincare plot data saved.
                                Defaults to 0.0.
        niter (int, optional): Number of toroidal periods in tracing. Defaults to 100.
        nfp (int, optional): Number of field periodicity. Defaults to 1.
        nstep (int, optional): Number of intermediate step for one period. Defaults to 1.

    Returns:
        array_like: The stored poincare date, shape is (len(r0), niter+1, 2).
    """
    from scipy.integrate import solve_ivp

    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))
        mag_rpz = np.array(
            [
                mag_xyz[0] * cosphi + mag_xyz[1] * sinphi,
                (-mag_xyz[0] * sinphi + mag_xyz[1] * cosphi) / rpz[0],
                mag_xyz[2],
            ]
        )
        return [mag_rpz[0] / mag_rpz[1], mag_rpz[2] / mag_rpz[1]]

    # some settings
    print("Begin field-line tracing: ")
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"})  # using LSODE
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-6})  # minimum tolerance
    # begin tracing
    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []
    for i in range(nlines):  # loop over each field-line
        points = [[r0[i], z0[i]]]
        for j in range(niter):  # loop over each toroidal iteration
            print_progress(i * niter + j + 1, nlines * niter)
            rz = points[j]
            phi_start = phi[j]
            for k in range(nstep):  # loop inside one iteration
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz, **kwargs)
                rz = sol.y[:, -1]
                phi_start += dphi
            points.append(rz)
        lines.append(np.array(points))
    return np.array(lines)


def tracing(bfield, r0, z0, phi0=0.0, niter=100, nfp=1, nstep=1,rtol=1e-8, **kwargs):
    """Trace magnetic field line in toroidal geometry

    Args:
        bfield (callable): A callable function.
                          The calling signature is `B = bfield(xyz)`, where `xyz`
                          is the position in cartesian coordinates and `B` is the
                          magnetic field at this point (in cartesian coordinates).
        r0 (list): Initial radial coordinates.
        z0 (list): Initial vertical coordinates.
        phi0 (float, optional): The toroidal angle where the poincare plot data saved.
                                Defaults to 0.0.
        niter (int, optional): Number of toroidal periods in tracing. Defaults to 100.
        nfp (int, optional): Number of field periodicity. Defaults to 1.
        nstep (int, optional): Number of intermediate step for one period. Defaults to 1.

    Returns:
        array_like: The stored poincare date, shape is (len(r0), niter+1, 2).
    """
    from scipy.integrate import solve_ivp

    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))
        mag_rpz = np.array(
            [
                mag_xyz[0] * cosphi + mag_xyz[1] * sinphi,
                (-mag_xyz[0] * sinphi + mag_xyz[1] * cosphi) / rpz[0],
                mag_xyz[2],
            ]
        )
        return [mag_rpz[0] / mag_rpz[1], mag_rpz[2] / mag_rpz[1]]

    # some settings
    print("Begin field-line tracing: ")
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"})  # using LSODE
    # begin tracing
    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []
    for i in range(nlines):  # loop over each field-line
        points = [[r0[i], z0[i]]]
        for j in range(niter):  # loop over each toroidal iteration
            print_progress(i * niter + j + 1, nlines * niter)
            rz = points[j]
            phi_start = phi[j]
            for k in range(nstep):  # loop inside one iteration
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz, **kwargs)
                rz = sol.y[:, -1]
                phi_start += dphi
            points.append(rz)
        lines.append(np.array(points))
    return np.array(lines)

def tracingplot(
    bfield, r0, z0,
    phi0=0.0, niter=100, nfp=1, nstep=1,
    # 新增简单参数
    plot=False,           # 是否绘图
    show=True,            # 是否展示窗口
    save=False,           # 是否保存
    save_dir=".",         # 保存目录
    save_prefix="tracing",# 前缀
    save_ext="png",       # 扩展名
    save_idx=1,           # 序号，由调用者传入
    dpi=150,              # 保存分辨率
    **kwargs
):
    """
    Trace magnetic field line in toroidal geometry.
    返回：np.array(lines) 形状 (len(r0), niter+1, 2)。
    """
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        xyz = np.array([rpz[0]*cosphi, rpz[0]*sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))
        mag_rpz = np.array([
            mag_xyz[0]*cosphi + mag_xyz[1]*sinphi,
            (-mag_xyz[0]*sinphi + mag_xyz[1]*cosphi)/rpz[0],
            mag_xyz[2]
        ])
        return [mag_rpz[0]/mag_rpz[1], mag_rpz[2]/mag_rpz[1]]

    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"})
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-6})

    dphi = 2*np.pi/nfp/nstep
    phi = phi0 + dphi*nstep*np.arange(niter)
    nlines = len(r0)
    lines = []
    for i in range(nlines):
        points = [[r0[i], z0[i]]]
        for j in range(niter):
            rz = points[j]
            phi_start = phi[j]
            for _ in range(nstep):
                sol = solve_ivp(fieldline, (phi_start, phi_start+dphi), rz, **kwargs)
                rz = sol.y[:, -1]
                phi_start += dphi
            points.append(rz)
        lines.append(np.array(points))

    lines = np.array(lines)

    # 绘图/保存
    if plot:
        fig, ax = plt.subplots()
        for i in range(len(lines)):
            ax.plot(lines[i][:,0], lines[i][:,1], label=f"line {i}")
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title("Tracing result")
        ax.legend()
        ax.grid(True)

        if save:
            fname = f"{save_prefix}_{save_idx:03d}.{save_ext}"
            import os
            outpath = os.path.join(save_dir, fname)
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
            print(f"Saved figure: {outpath}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return lines


def tracingFULL(bfield, r0, z0, phi0=0.0, order=10,niter=100, nfp=1, nstep=1,FullLine=True, **kwargs):
    """Trace magnetic field line in toroidal geometry

    Args:
        bfield (callable): A callable function.
                          The calling signature is `B = bfield(xyz)`, where `xyz`
                          is the position in cartesian coordinates and `B` is the
                          magnetic field at this point (in cartesian coordinates).
        r0 (list): Initial radial coordinates.
        z0 (list): Initial vertical coordinates.
        phi0 (float, optional): The toroidal angle where the poincare plot data saved.
                                Defaults to 0.0.
        niter (int, optional): Number of toroidal periods in tracing. Defaults to 100.
        nfp (int, optional): Number of field periodicity. Defaults to 1.
        nstep (int, optional): Number of intermediate step for one period. Defaults to 1.
        FullLine:If true, it will return lines, each has nstep+1 points
    Returns:
        array_like: The stored poincare date, shape is (len(r0), niter+1, 2).
    """
    from scipy.integrate import solve_ivp

    # define the integrand in cylindrical coordinates
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xyz = np.array([rpz[0] * cosphi, rpz[0] * sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))
        
        # Cylindrical B field components
        try:
            Br = mag_xyz[0] * cosphi + mag_xyz[1] * sinphi
            Bphi = (-mag_xyz[0] * sinphi + mag_xyz[1] * cosphi) / rpz[0]
            Bz = mag_xyz[2]
        except ZeroDivisionError:
            print(f"rpz[0] = 0 at phi = {phi}")
            return [0.0, 0.0]

        eps = 1e-12
        if not np.isfinite(Bphi) or abs(Bphi) < eps:
            # print(f"Invalid Bphi = {Bphi} at phi = {phi}, xyz = {xyz}")
            return [0.0, 0.0]

        return [Br / Bphi, Bz / Bphi]


    # some settings
    print("Begin field-line tracing: ")
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"})  # using LSODE
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-6})  # minimum tolerance
    # begin tracing
    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []
    for i in range(nlines):  # loop over each field-line
        points = [[r0[i], z0[i]]]
        po=[[r0[i], z0[i],phi[i]]]
        for j in range(niter):  # loop over each toroidal iteration
            print_progress(i * niter + j + 1, nlines * niter)
            rz = points[j]
            phi_start = phi[j]
            for k in range(nstep):  # loop inside one iteration
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz, **kwargs)
                rz = sol.y[:, -1]
                phi_start += dphi
                po.append([rz[0],rz[1],phi_start])
                points.append(rz)
        if FullLine:
            lines.append(np.array(po))
        else:
            lines.append(np.array(points))
        # lines=np.array(lines)
        # curve=rzp2curverz(lines=lines,order=order )
    lines=np.array(lines)
    return lines



def loadcurve(filename='rz.npz'):
    data = np.load(filename)
    (r_c,z_s)=(data['r_c'],data['z_s'])
    order=len(r_c)-1
    curve=CurveRZFourier(quadpoints=361, order=order, nfp=1, stellsym=0)
    curve.rc[:] = r_c[:order+1]
    curve.zs[:] = z_s[:order+1]
    return curve


def fullaxplot(coils, rz0=None, phi0=0, bounds=None,
           order=10, niter=1, nstep=10,
           save=False, save_dir=".", prefix="fullax", **kwargs):
    """
    优化磁轴，并可在优化过程中自动保存每次 tracing 图像。
    """
    if rz0 is None:
        rz0 = [findrz0(coils), 0]

    def field(pos):
        b = 0
        for i in range(len(coils)):
            b += coils.data[i].bfield_HH(pos)
        return b

    # 调用计数器
    call_counter = {"n": 0}

    def fun(rz):
        if not np.all(np.isfinite(rz)) or rz[0] <= 0:
            print(f"rz 非法: {rz}")
            return 1e6

        try:
            lines = tracing(field, [rz[0]], [rz[1]],
                            niter=niter, phi0=phi0, nstep=nstep, **kwargs)
            if not np.all(np.isfinite(lines)):
                print(f"tracing() 返回 nan, rz = {rz}")
                return 1e6

            line = lines[0]
            r_vals = [pt[0] for pt in line]
            z_vals = [pt[1] for pt in line]

            # 自动保存
            if save:
                call_counter["n"] += 1
                idx = call_counter["n"]
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"{prefix}_{idx:03d}.png")

                plt.figure()
                plt.plot(r_vals, z_vals, label=f"rz = {rz}")
                plt.xlabel("r")
                plt.ylabel("z")
                plt.title(f"Tracing Line for rz = {rz}")
                plt.legend()
                plt.grid(True)
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved: {fname}")

            # 目标函数值
            r = (line[1][0] - rz[0])**2 + (line[1][1] - rz[1])**2
            return r

        except Exception as e:
            print(f"tracing 失败, rz = {rz}, 错误: {e}")
            return 1e6

    res = minimize(fun, rz0, bounds=bounds)
    print(res)

    axlist = tracingFULL(field, [res.x[0]], [res.x[1]],
                         niter=1, nstep=360, phi0=phi0, FullLine=1, **kwargs)
    ma = rzp2curverz(axlist)
    return ma



def fullax(coils,rz0=None,phi0=0,bounds=None,order=10,niter=1, nstep=10,rtol=1e-8,**kwargs):
    #if bounds is None:
        #bounds=[(rz0[0]-0.5,rz0[0]+0.5),(-0.1,0.1)]
    if rz0==None:
        rz0=[findrz0(coils),0]
    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=coils.data[i].bfield_HH(pos)
        # print(pos,'->',b)
        return b
    def fun(rz):
        # 检查是否包含 nan/inf 或非法初始半径
        if not np.all(np.isfinite(rz)) or rz[0] <= 0:
            print(f"rz 非法: {rz}")
            return 1e6  # 惩罚项，优化器会避开这组参数

        try:
            lines = tracing(field, [rz[0]], [rz[1]], niter=niter, phi0=phi0, nstep=nstep,rtol=rtol,**kwargs)
            if not np.all(np.isfinite(lines)):
                print(f"tracing() 返回包含 nan 的轨迹, rz = {rz}")
                return 1e6
            line = lines[0]

            r = (line[1][0] - rz[0])**2 + (line[1][1] - rz[1])**2
            return r
            #print(lines)
            #print(lines[0][1][0],rz[0],lines[0][1][1],rz[1])
            #print(r)
        except Exception as e:
            print(f"tracing 失败, rz = {rz}，错误: {e}")
            return 1e6


    res=minimize(fun,rz0,bounds=bounds)
    print(res)

    axlist=tracingFULL(field,[res.x[0]],[res.x[1]],niter=1,nstep=360,phi0=phi0,FullLine=1,**kwargs)
    # print(type(axlist))
    # print(axlist)
    ma=rzp2curverz(axlist)
    return ma



def findax(coils,rz0=None,phi0=0,bounds=None,**kwargs):
    if bounds is None:
        bounds=[(rz0[0]-0.5,rz0[0]+0.5),(-0.1,0.1)]
    if rz0==None:
        rz0=[findrz0(coils),0]
    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=coils.data[i].bfield_HH(pos)
        return b
    def fun(rz):
        lines=tracing(field,[rz[0]],[rz[1]],niter=1,phi0=phi0,**kwargs)
        r=(lines[0][1][0]-rz[0])**2+(lines[0][1][1]-rz[1])**2
        #print(lines)
        #print(lines[0][1][0],rz[0],lines[0][1][1],rz[1])
        #print(r)
        lines=None
        return r
    res=minimize(fun,rz0,method='L-BFGS-B',bounds=bounds)
    print(res)
    return res.x

def trace_fieldline_wout(filename):
    vmec = Vmec(filename)
    keys = vmec.wout.__dict__.keys()

    ns   = vmec.wout.ns
    nfp  = vmec.wout.nfp
    #nfp = 1
    s = -1
    bsubu = vmec.wout.bsupumnc[:,s]
    bsubv = vmec.wout.bsupvmnc[:,s]
    bmnc  = vmec.wout.bmnc[:,s]
    rmnc  = vmec.wout.rmnc[:,s]
    zmns  = vmec.wout.zmns[:,s]
    raxis_cc = vmec.wout.raxis_cc


    # 对于形状的傅里叶展开系数
    xm    = vmec.wout.xm
    xn    = vmec.wout.xn

    # 对于B field的Nyquist频率
    xm_nyq = vmec.wout.xm_nyq
    xn_nyq = vmec.wout.xn_nyq


    # 获得最外层磁面的磁场大小
    def B_surface():
        B_surface = 0
        theta = np.linspace(0,2*np.pi,1000)
        zeta  = np.linspace(0,2*np.pi,1000)
        t2,z2 = np.meshgrid(theta,zeta)
        nfp = 1
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * t2 - n * nfp * z2
            B_surface += bmnc[jmn] * np.cos(angle)
        return B_surface.T



    R_2D = np.zeros((1000 , 1000 ))
    Z_2D = np.zeros((1000 , 1000 ))
    phi1d   = np.linspace(0, 2*np.pi,1000   ,endpoint=True)
    theta1d = np.linspace(0, 2*np.pi,1000   ,endpoint=True)
    p2,t2   = np.meshgrid(phi1d,theta1d)
    for jmn in range(len(xm)):
            m = xm[jmn]
            n = xn[jmn]
            angle = m * t2 - n * p2
            R_2D += rmnc[jmn] * np.cos(angle)
            Z_2D += zmns[jmn] * np.sin(angle)

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2D * np.cos(phi1d)
    y_2D_plot = R_2D * np.sin(phi1d)
    z_2D_plot = Z_2D

def run_fieldline_tracing(curves, currents, ma, s, nfp=3, nfieldlines=10, tmax_fl=20000, degree=4, out_dir="./output/", save_data=True):
    #输入都是simsopt, ma可以尝试由fullax找
    os.makedirs(out_dir, exist_ok=True)

    # Setup coils and magnetic field
    coils = coils_via_symmetries(curves, currents, nfp, True)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
    print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

    # Setup surface classifier
    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)

    # Create interpolation grid bounds
    n = 20
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    zrange = (0, np.max(zs), n//2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    def trace_fieldlines(bfield, label):
        t1 = time.time()
        R0 = np.linspace(ma.gamma()[0, 0], ma.gamma()[0, 0] + 0.14, nfieldlines)
        Z0 = [ma.gamma()[0, 2] for _ in range(nfieldlines)]
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-7,
            phis=phis,
            stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)]
        )
        t2 = time.time()
        print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum(len(l) for l in fieldlines_tys) // nfieldlines}", flush=True)
        plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(out_dir, f'poincare_fieldline_{label}.png'), dpi=150)
        
        if save_data:
            np.savez(os.path.join(out_dir, f'fieldlines_{label}.npz'),
                     tys=fieldlines_tys,
                     phi_hits=fieldlines_phi_hits,
                     R0=R0,
                     Z0=Z0,
                     phis=phis)
            print(f"Saved fieldline data to {os.path.join(out_dir, f'fieldlines_{label}.npz')}")

    # Interpolated field setup
    print('Initializing InterpolatedField')
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
    )
    print('Done initializing InterpolatedField')

    bsh.set_points(ma.gamma().reshape((-1, 3)))
    bs.set_points(ma.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    print("|B-Bh| on axis", np.sort(np.abs(B - Bh).flatten()))

    # Trace field lines
    print('Beginning field line tracing')
    trace_fieldlines(bsh, 'bsh')


def poincareplot(coils,phi0=0,rz0=[1,0],len=0.4,nfieldlines=8,point_num=100,s=None,ma=None,show=False,save=False,**kwargs):
    #输入是coilpy线圈
    if ma is None:
        ma=findax(coils,rz0=rz0,phi0=phi0)#,**kwargs
    r0 = np.linspace(ma[0], ma[0]+len, nfieldlines)
    z0 = np.array([ma[1] for i in range(nfieldlines)]) 
    def field(pos):#B
        b=0
        for coil in coils:
            b+=coil.bfield_HH(pos)
        return b
    poincare_plot(tracing(field,r0,z0,phi0=phi0,niter=point_num,**kwargs))
    #plt.xlim(1,2)
    #plt.ylim(-0.8,0.8)
    if s is not None:
        from coilpy.surface import FourSurf
        s=s.to_RZFourier()
        s.write_nml('temp.nml')
        nml_to_focus("temp.nml", "temp.boundary", nfp=s.nfp)
        boundary=FourSurf.read_focus_input("temp.boundary")
        boundary.plot(zeta=0, label='temp.boundary', color='k', linestyle='--')
    if save:
        plt.savefig('poincare_plot.png')#,clip_on=False
    if show:
        plt.show()
    print('success')




if __name__ == "__main__":
    from simsopt._core import load, save
    from simsopt.field import coils_to_focus, BiotSavart
    from simsopt.geo import SurfaceRZFourier,CurveXYZFourier    
    ID = 958# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000  
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')
    cpcoils=from_simsopt(coils)
    ma=fullax(cpcoils,rz0=[0.876,0])
    plotting.plot([ma])





    from simsopt._core import load, save
    from simsopt.field import coils_to_focus, BiotSavart
    from simsopt.geo import SurfaceRZFourier,CurveXYZFourier
    ID = 400049# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000  
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')
    cpcoils=from_simsopt(coils)
    poincareplot(cpcoils,rz0=[0.8763,0],len=0.1,show=True)#磁轴位置x: [ 8.763e-01  4.552e-06]


    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=cpcoils.data[i].bfield_HH(pos)
        # print(pos,'->',b)
        return b

    axlist=tracingFULL(field,[8.763e-01],[4.552e-06],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    ma=rzp2curverz(axlist)

    axlist2=tracingFULL(field,[0.9],[0.3],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line2=rzp2curverz(axlist2)

    axlist3=tracingFULL(field,[0.86],[-0.3],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line3=rzp2curverz(axlist3)


    plotting.plot([line3,line2,ma])


    # from coilpy.surface import FourSurf
    # surf=surfaces[-1]
    # surf=surf.to_RZFourier()
    # #surf.change_resolution(12,12)
    # surf.write_nml('temp.nml')

    # nml_to_focus("temp.nml", "temp.boundary", nfp=surf.nfp)
    # boundary=FourSurf.read_focus_input("temp.boundary")



    # import plotly.graph_objects as go
    # fig = go.Figure()
    # boundary.plot3d(engine='plotly', fig=fig,show=False)
    # cpcoils.plot(engine='plotly', fig=fig,show=False)

    # fig.update_layout(
    #     scene=dict(
    #         aspectmode='data',
    #         xaxis = dict(showbackground=False,visible=False),
    #         yaxis = dict(showbackground=False,visible=False),
    #         zaxis = dict(showbackground=False,visible=False),
    #         bgcolor='rgba(0,0,0,0)',
    #     )
    # )

    # fig.write_html('focus.html')





