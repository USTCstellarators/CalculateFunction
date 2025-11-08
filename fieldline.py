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
import os, re
from simsopt.field import (InterpolatedField, coils_via_symmetries, SurfaceClassifier,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.field import BiotSavart as simsopt_BiotSavart
import time
from simsopt.mhd import Vmec
from mymisc import rzp2curverz,from_simsopt,nml_to_focus,rzp2curverznfp,rzphi_to_xyz,coil2rz0

def findrz0(coils):
    rz0 = np.mean([
        np.sqrt(d.x**2 +d.y**2)
        for d in coils.data
    ])
    print('rz0 =', rz0)
    return rz0

def save_line_image(line_rzphi):
    """
    自动保存磁力线图像：
      - 保存到 frames/optK/coil_xxx.png
      - 每次程序运行自动找到 frames 下最大的 optN，新建 opt{N+1}
      - coil_xxx 自动按顺序递增
    """
    base_dir = "frames"
    os.makedirs(base_dir, exist_ok=True)

    # ---- 第一次调用时自动确定本次运行目录 ----
    if not hasattr(save_line_image, "_run_dir"):
        existing = []
        for name in os.listdir(base_dir):
            m = re.fullmatch(r"opt(\d+)", name)
            if m:
                existing.append(int(m.group(1)))
        next_k = (max(existing) + 1) if existing else 0
        run_dir = os.path.join(base_dir, f"opt{next_k}")
        os.makedirs(run_dir, exist_ok=True)

        save_line_image._run_dir = run_dir
        save_line_image._coil_idx = 0

    # ---- 生成保存路径 ----
    idx = save_line_image._coil_idx
    path = os.path.join(save_line_image._run_dir, f"coil_{idx:03d}.png")

    # ---- 绘图 ----
    r = line_rzphi[:, 0]
    z = line_rzphi[:, 1]
    phi = line_rzphi[:, 2]
    x, y, zc = rzphi_to_xyz(r, z, phi)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, zc, lw=1.2, color='royalblue', alpha=0.9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Magnetic Field Line,rz=[{r[0]},{z[0]}]", fontsize=12)
    ax.view_init(elev=25, azim=-60)
    ax.grid(True)

    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ---- 更新计数器 ----
    save_line_image._coil_idx += 1
    return path





def tracing(
    bfield, r0, z0, phi0=0.0,
    niter=1, nfp=1, nstep=1,
    rtol=1e-8, method='BDF'
):
    """
    轻量版磁力线追踪：
      - 仅返回 (len(r0), niter+1, 2): [r, z]
    """
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        xyz = np.array([rpz[0]*cosphi, rpz[0]*sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))

        try:
            Br = mag_xyz[0]*cosphi + mag_xyz[1]*sinphi
            Bphi = (-mag_xyz[0]*sinphi + mag_xyz[1]*cosphi) / rpz[0]
            Bz = mag_xyz[2]
        except ZeroDivisionError:
            return [0.0, 0.0]

        eps = 1e-10
        if not np.isfinite(Bphi) or abs(Bphi) < eps:
            return [0.0, 0.0]
        return [Br/Bphi, Bz/Bphi]

    # 设置积分步长
    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []

    for i in range(nlines):
        points = [[r0[i], z0[i]]]  # 只保存每圈末点
        rz = [r0[i], z0[i]]
        for j in range(niter):
            phi_start = phi[j]
            for _ in range(nstep):
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz,
                                method=method, rtol=rtol)
                rz = sol.y[:, -1]
                phi_start += dphi
            points.append(rz)  # 只存每圈末点
        lines.append(np.array(points))

    return np.array(lines)




def tracingFULL(
    bfield, r0, z0, phi0=0.0,
    order=10, niter=100, nfp=1, nstep=1,
    FullLine=True, show_progress=False,
    plot=False, save_dir="frames",rtol=1e-8,method='BDF'
):
    """Trace magnetic field line in toroidal geometry.
    
    Args:
        FullLine (bool): True=保存完整轨迹 (r,z,phi)，False=每圈一个点 (r,z)
        plot (bool): 是否为每条线保存图像
    """
    def fieldline(phi, rz):
        rpz = np.array([rz[0], phi, rz[1]])
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        xyz = np.array([rpz[0]*cosphi, rpz[0]*sinphi, rpz[2]])
        mag_xyz = np.ravel(bfield(xyz))

        try:
            Br = mag_xyz[0]*cosphi + mag_xyz[1]*sinphi
            Bphi = (-mag_xyz[0]*sinphi + mag_xyz[1]*cosphi) / rpz[0]
            Bz = mag_xyz[2]
        except ZeroDivisionError:
            return [0.0, 0.0]

        eps = 1e-10
        if not np.isfinite(Bphi) or abs(Bphi) < eps:
            return [0.0, 0.0]
        return [Br/Bphi, Bz/Bphi]



    dphi = 2 * np.pi / nfp / nstep
    phi = phi0 + dphi * nstep * np.arange(niter)
    nlines = len(r0)
    lines = []

    for i in range(nlines):
        points = [[r0[i], z0[i]]]
        po = [[r0[i], z0[i], phi0]]
        for j in range(niter):
            if show_progress:
                print(f"[{i+1}/{nlines}] period {j+1}/{niter}")
            rz = points[-1]
            phi_start = phi[j]
            for _ in range(nstep):
                sol = solve_ivp(fieldline, (phi_start, phi_start + dphi), rz)
                rz = sol.y[:, -1]
                phi_start += dphi
                po.append([rz[0], rz[1], phi_start])
                points.append(rz)

        po = np.array(po)
        if FullLine:
            line = po  # (niter*nstep+1, 3)
        else:
            # 只取每圈结束点（含起点），每圈有 nstep 步
            keep = po[::nstep, :]
            line = keep[:, :2]  # (niter+1, 2)
        lines.append(line)

        if plot and FullLine:
            save_line_image(po)   


    return np.array(lines)


def fullax(coils,rz0=None,phi0=0,nfp=1,bounds=None,order=10,niter=1, nstep=10,method='BDF',rtol=1e-6,plot=False):
    # if bounds is None:
    #     bounds=[(rz0[0]-0.5,rz0[0]+  0.5),(-0.01,0.01)]

    res=findax(coils,rz0=rz0,phi0=phi0,nfp=nfp,bounds=bounds,rtol=rtol,method=method)
    ##for coilpy coils
    coils=from_simsopt(coils)
    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=coils.data[i].bfield_HH(pos)
        # print(pos,'->',b)
        return b

    ##for simsopt coils
    # def field(pos):
    #     fl=simsopt_BiotSavart(coils)
    #     fl.set_points(np.array([pos]))
    #     b=fl.B()[0]
    #     return b
    axlist=tracingFULL(field,[res[0]],[res[1]],niter=1,nstep=360,phi0=phi0,FullLine=1)
    # print(type(axlist))
    # print(axlist)
    ma=rzp2curverz(axlist)
    return ma




def findax(coils,rz0=None,phi0=0,nfp=1,bounds=None,rtol=1e-6,method='BDF',plot=False):

    coils=from_simsopt(coils)
    if rz0==None:
        rz0=[findrz0(coils),0]
    if bounds is None:
        bounds=[(rz0[0]-0.2,rz0[0]+0.2),(-0.001,0.001)]
    ##for coilpy coils
    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=coils.data[i].bfield_HH(pos)
        # print(pos,'->',b)
        return b

    ##for simsopt coils
    # def field(pos):
    #     fl=simsopt_BiotSavart(coils)
    #     fl.set_points(np.array([pos]))
    #     b=fl.B()[0]
    #     return b
    def fun(rz):
        # lines=tracingFULL(field,[rz[0]],[rz[1]],niter=1,nstep=360,phi0=phi0,rtol=rtol,plot=plot,**kwargs)
        lines=tracing(field,[rz[0]],[rz[1]],niter=1,nfp=nfp,phi0=phi0,method=method,rtol=rtol)
        # if plot:
            # lines=tracingFULL(field,[rz[0]],[rz[1]],niter=1,nstep=360,phi0=phi0,plot=plot,**kwargs)
        # else:
        #     lines=tracing(field,[rz[0]],[rz[1]],niter=1,phi0=phi0,**kwargs)
        r=(lines[0][-1][0]-rz[0])**2+(lines[0][-1][1]-rz[1])**2
        #print(lines)
        #print(lines[0][1][0],rz[0],lines[0][1][1],rz[1])
        #print(r)
        lines=None
        return r
    res=minimize(fun,rz0,method='powell',bounds=bounds)
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
    poincare_plot(tracing(field,r0,z0,phi0=phi0,niter=point_num,show_progress=True,**kwargs))
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

    import numpy as np
    np.set_printoptions(threshold=np.inf)
    from simsopt._core import load, save
    from simsopt.field import coils_to_focus, BiotSavart
    from simsopt.geo import SurfaceRZFourier,CurveXYZFourier,plotting
    import matplotlib.pyplot as plt
    ID = 400049# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000  
    [surfaces, coils] = load(f'../../projects/QUASR_08072024/simsopt_serials/{fID:04}/serial{ID:07}.json')
    x=surfaces[0].gamma()[0,:,0]
    y=surfaces[0].gamma()[0,:,1]

    r0=np.mean(np.sqrt(x**2+y**2))
    print(r0)
    cpcoils=from_simsopt(coils)

    ma=fullax(cpcoils,rz0=[r0,0],nfp=surfaces[-1].nfp,plot=1,method='BDF')
    # plotting.plot([ma])

    # poincareplot(cpcoils,rz0=[0.8763,0],ma=[0.8763,0],len=0.1,show=True)#磁轴位置x: [ 8.763e-01  4.552e-06]












    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=cpcoils.data[i].bfield_HH(pos)
        # print(pos,'->',b)
        return b

    axlist1=tracingFULL(field,[8.763e-01],[0],niter=1,nstep=360)
    # print(type(axlist))
    print(axlist1)
    ma=rzp2curverz(axlist1)

    axlist2=tracingFULL(field,[r0],[0],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line2=rzp2curverz(axlist2)

    axlist3=tracingFULL(field,[0.86],[0],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line3=rzp2curverz(axlist3)


    # plotting.plot([line3,line2,ma])



    x1, y1, z1 =rzphi_to_xyz(axlist1[0,:,0], axlist1[0,:,1], axlist1[0,:,2])
    x2, y2, z2 =rzphi_to_xyz(axlist2[0,:,0], axlist2[0,:,1], axlist2[0,:,2])
    x3, y3, z3 =rzphi_to_xyz(axlist3[0,:,0], axlist3[0,:,1], axlist3[0,:,2])



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, y1, z1, c='red', s=30, alpha=0.6)
    ax.scatter(x2, y2, z2, c='green', s=30, alpha=0.6)
    ax.scatter(x3, y3, z3, c='blue', s=30, alpha=0.6)


    plt.show()