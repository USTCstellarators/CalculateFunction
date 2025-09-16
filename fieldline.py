import numpy as np
from math import pi
import matplotlib.pyplot as plt
from coilpy import Coil as cl
from coilpy.misc import poincare_plot,print_progress,tracing
from scipy.optimize import minimize
from simsopt.geo.curverzfourier import CurveRZFourier
from numpy import arctan
from mpl_toolkits.mplot3d import Axes3D
from simsopt.geo import CurveXYZFourier
from scipy.optimize import curve_fit

def nml_to_focus(nml_filename, focus_filename, nfp=2):
    import re
    """
    从 VMEC .nml 文件中提取 RBC/ZBS 的 (n, m) 项，自动计算 bmn,写入 FOCUS 所需的 .boundary 文件。

    参数:
        nml_filename: str, 输入的 .nml 文件路径
        focus_filename: str, 输出的 .boundary 文件路径
        nfp (int): 场周期数
    """
    with open(nml_filename, 'r') as f:
        text = f.read()

    # 提取 NFP 值
    nfp_match = re.search(r'NFP\s*=\s*(\d+)', text, re.IGNORECASE)
    if not nfp_match:
        raise ValueError("NFP 未在 NML 文件中找到。")
    nfp = int(nfp_match.group(1))

    # 正则提取所有 RBC 和 ZBS 项 (n, m)
    rbc_pattern = re.findall(r'RBC\(\s*(-?\d+)\s*,\s*(\d+)\s*\)\s*=\s*([Ee0-9\+\-\.]+)', text)
    zbs_pattern = re.findall(r'ZBS\(\s*(-?\d+)\s*,\s*(\d+)\s*\)\s*=\s*([Ee0-9\+\-\.]+)', text)

    # 构建 {(n, m): value} 字典
    rbc_dict = {(int(n), int(m)): float(val) for n, m, val in rbc_pattern}
    zbs_dict = {(int(n), int(m)): float(val) for n, m, val in zbs_pattern}

    # 所有键组合
    all_keys = sorted(set(rbc_dict.keys()) | set(zbs_dict.keys()), key=lambda x: (x[1], x[0]))

    bmn = len(all_keys)

    # 写入 .boundary 文件
    with open(focus_filename, 'w') as f:
        f.write("# bmn   bNfp   nbf\n")
        f.write(f"{bmn:3d} \t {nfp} \t 0\n")
        f.write("# Plasma boundary\n")
        f.write("# n m Rbc Rbs Zbc Zbs\n")
        for n, m in all_keys:
            rbc = rbc_dict.get((n, m), 0.0)
            zbs = zbs_dict.get((n, m), 0.0)
            f.write(f"{n:5d} {m:5d} {rbc: .15E}  0.000000000000000E+00  0.000000000000000E+00  {zbs: .15E}\n")

    print(f" 成功将 {bmn} 项输出至 {focus_filename}(NFP = {nfp})")

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


def xyzp2curvexyz(points, order=10, nfp=1):
    """
    给定一个 3D 点序列（表示一条闭合线圈），拟合其傅里叶系数并生成 CurveXYZFourier 对象。
    
    Args:
        points: 3D 坐标序列，形状为 [n_points, 3] 的列表或数组
        order: 傅里叶展开的最高阶数
        nfp: 场周期数
    
    Returns:
        CurveXYZFourier 对象，包含拟合后的傅里叶系数
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be a (N, 3) array.")
    
    npts = len(points)
    phi = np.linspace(0, 2 * np.pi / nfp, npts, endpoint=False)
    x_vals, y_vals, z_vals = points[:, 0], points[:, 1], points[:, 2]
    
    def fourier_cos_series(phi, *a):
        return sum(a[m] * np.cos(nfp * m * phi) for m in range(len(a)))

    def fourier_sin_series(phi, *a):
        return sum(a[m] * np.sin(nfp * (m+1) * phi) for m in range(len(a)))

    def fit_series(vals, kind='cos'):
        p0 = np.zeros(order + 1)
        if kind == 'cos':
            params, _ = curve_fit(fourier_cos_series, phi, vals, p0=p0)
        elif kind == 'sin':
            params, _ = curve_fit(fourier_sin_series, phi, vals, p0=p0[:-1])
            params = np.insert(params, 0, 0.0)  # z_s[0] = 0 for sin series
        return params

    x_c = fit_series(x_vals, kind='cos')
    y_c = fit_series(y_vals, kind='cos')
    z_s = fit_series(z_vals, kind='sin')

    # 构造 CurveXYZFourier 对象
    curve = CurveXYZFourier(quadpoints=npts, order=order )
    curve.xc[:] = x_c[:order + 1]
    curve.yc[:] = y_c[:order + 1]
    curve.zs[:] = z_s[:order + 1]
    curve.x = curve.get_dofs()

    return curve

def rzp2curverz(lines,order=10):
    if not isinstance(lines, (list, np.ndarray)) or len(lines) == 0:
        raise ValueError("lines must be a non-empty list of fieldline point arrays.")

    def rz_cofficients(r_vals,z_vals,order,nfp):
        npoint=len(r_vals)
        from scipy.optimize import curve_fit
        phi = np.linspace(0, 2 * np.pi/nfp, npoint)
        def R_fourier_series(phi,*a):
            n_terms = len(a)
            n_fp = nfp   
            result=0
            for m in range(n_terms):
                result+= a[m] * np.cos(n_fp * m * phi)
            return result

        def Z_fourier_series(phi,*a):
            n_terms = len(a)
            n_fp = nfp
            result=0
            for m in range(1,n_terms):
                result += a[m] * np.sin(n_fp * m * phi)
            return result

        def fit_fourier_series(phi, R, Z, order):
            initial_guess = np.zeros(order)

            r_params, _ = curve_fit(R_fourier_series, phi, R, p0=initial_guess)
            z_params, _ = curve_fit(Z_fourier_series, phi, Z, p0=initial_guess)

            r_c = r_params
            z_s = z_params[1:]

            return r_c, z_s

        r_c, z_s = fit_fourier_series(phi, r_vals, z_vals, order)
        return r_c,z_s
    curve=CurveRZFourier(quadpoints=len(lines[0]), order=order, nfp=1, stellsym=0)
    #ma.least_squares_fit(axlist[0])
    (r_c,z_s)=rz_cofficients(np.array([p[0] for p in lines[0]]),np.array([p[1] for p in lines[0]]),order=order+1,nfp=1)
    curve.rc[:] = r_c[:order+1]
    curve.zs[:] = z_s[:order+1]
    curve.x = curve.get_dofs()
    return curve








def points2surfacexyz(points_xyz, mpol=10, ntor=10, nfp=1, stellsym=False):
    """
    通过拟合输入的 3D 网格点来初始化 SurfaceXYZTensorFourier。
    不使用 least_squares_fit。
    """
    import numpy as np
    from simsopt.geo import SurfaceXYZTensorFourier
    from scipy.optimize import curve_fit
    points_xyz = np.asarray(points_xyz)
    nphi, ntheta, _ = points_xyz.shape

    theta_grid = np.linspace(0, 1.0, ntheta, endpoint=False)
    zeta_grid = np.linspace(0, 1.0/nfp, nphi, endpoint=False)

    Theta, Zeta = np.meshgrid(theta_grid, zeta_grid, indexing='ij')
    Theta = Theta.flatten()
    Zeta = Zeta.flatten()

    coords = {
        'x': points_xyz[:, :, 0].flatten(),
        'y': points_xyz[:, :, 1].flatten(),
        'z': points_xyz[:, :, 2].flatten()
    }

    def fourier_2d(theta, zeta, *coeffs):
        result = np.zeros_like(theta)
        idx = 0
        for m in range(mpol+1):
            for n in range(-ntor, ntor+1):
                result += coeffs[idx] * np.cos(2*np.pi*(m*theta - n*nfp*zeta))
                idx += 1
        for m in range(1, mpol+1):
            for n in range(-ntor, ntor+1):
                result += coeffs[idx] * np.sin(2*np.pi*(m*theta - n*nfp*zeta))
                idx += 1
        return result

    n_coeffs = (mpol+1)*(2*ntor+1) + mpol*(2*ntor+1)

    fitted_coeffs = {}
    for key in coords:
        popt, _ = curve_fit(lambda th, ze, *a: fourier_2d(th, ze, *a),
                            (Theta, Zeta), coords[key], p0=np.zeros(n_coeffs))
        fitted_coeffs[key] = popt

    # 创建 Surface
    surf = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym)
    print(surf.dof_names)
    # 赋值 dofs（曲面系数）
    dofs = []
    for key in ('x', 'y', 'z'):
        dofs.extend(fitted_coeffs[key])
    surf.set_dofs(dofs)

    return surf




def points2surfacerz(points_xyz, mpol=10, ntor=10, nfp=1, stellsym=True):
    """
    用 r,z 网格点拟合出 SurfaceRZFourier 曲面，不依赖 least_squares_fit。
    
    参数:
        r_vals, z_vals: ndarray, shape (ntheta, nzeta)
            在 (theta, zeta) 网格上的 R/Z 值
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from simsopt.geo import SurfaceRZFourier

    r_vals = np.sqrt(points_xyz[:,:, 0]**2 + points_xyz[:,:, 1]**2)
    z_vals = points_xyz[:,:,  2]


    ntheta, nzeta = r_vals.shape
    theta = np.linspace(0, 1.0, ntheta, endpoint=False)
    zeta = np.linspace(0, 1.0 / nfp, nzeta, endpoint=False)
    TH, ZE = np.meshgrid(theta, zeta, indexing='ij')

    def rz_fourier_basis(th, ze, *coeffs):
        result = np.zeros_like(th)
        idx = 0
        # cos(mθ - nζ) 部分
        for m in range(mpol+1):
            for n in range(-ntor, ntor+1):
                result += coeffs[idx] * np.cos(2*np.pi*(m*th - n*nfp*ze))
                idx += 1
        # sin(mθ - nζ) 部分
        for m in range(1, mpol+1):
            for n in range(-ntor, ntor+1):
                result += coeffs[idx] * np.sin(2*np.pi*(m*th - n*nfp*ze))
                idx += 1
        
        return result

    ncoeffs = (mpol+1)*(2*ntor+1) + mpol*(2*ntor+1)

    popt_r, _ = curve_fit(lambda th_ze, *a: rz_fourier_basis(th_ze[0], th_ze[1], *a),
                          (TH.flatten(), ZE.flatten()), r_vals.flatten(), p0=np.zeros(ncoeffs))
    
    popt_z, _ = curve_fit(lambda th_ze, *a: rz_fourier_basis(th_ze[0], th_ze[1], *a),
                          (TH.flatten(), ZE.flatten()), z_vals.flatten(), p0=np.zeros(ncoeffs))

    # 创建 surface
    surf = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym)
    
    # 拆解系数
    idx = 0
    for m in range(mpol+1):
        for n in range(-ntor, ntor+1):
            surf.set_rc(m, n,popt_r[idx])
            surf.set_zc(m, n,popt_z[idx])

            idx += 1
    for m in range(1, mpol+1):
        for n in range(-ntor, ntor+1):
            surf.set_rs(m, n,popt_r[idx])
            surf.set_zs(m, n,popt_z[idx])            

            idx += 1

    surf.x = surf.get_dofs()  # 可选

    return surf

def loadcurve(filename='rz.npz'):
    data = np.load(filename)
    (r_c,z_s)=(data['r_c'],data['z_s'])
    order=len(r_c)-1
    curve=CurveRZFourier(quadpoints=361, order=order, nfp=1, stellsym=0)
    curve.rc[:] = r_c[:order+1]
    curve.zs[:] = z_s[:order+1]
    return curve




def fullax(coils,rz0=[1,0],phi0=0,bounds=None,order=10,**kwargs):
    #if bounds is None:
        #bounds=[(rz0[0]-0.5,rz0[0]+0.5),(-0.1,0.1)]
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
            lines = tracing(field, [rz[0]], [rz[1]], niter=1, phi0=phi0, **kwargs)
            if not np.all(np.isfinite(lines)):
                print(f"tracing() 返回包含 nan 的轨迹, rz = {rz}")
                return 1e6
            r = (lines[0][1][0]-rz[0])**2 + (lines[0][1][1]-rz[1])**2
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


def findax_safe(coils,rz0=[[1,0],[0.5,0],[1.5,0]],phi0=0,bounds=None,**kwargs):
    if bounds is None:
        bounds=[(rz0[0]-0.5,rz0[0]+0.5),(-0.1,0.1)]

    # ---- 新增：安全封装，抓取 LSODA 的 "t+h=t" 告警 ----
    import io, numpy as np
    from contextlib import redirect_stdout
    class _TraceStalled(RuntimeError):
        pass
    def _safe_tracing(field, R, Z, phi0, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            lines = tracing(field, [R], [Z], niter=1, phi0=phi0, **kw)
        out = buf.getvalue()
        if ("lsoda--  warning..internal t (=r1) and h (=r2)" in out) or ("t + h = t" in out):
            raise _TraceStalled("LSODA step-size underflow (t+h==t).")
        return lines
    # ---------------------------------------------------

    def field(pos):
        b=0
        for i in range(len(coils)):
            b+=coils.data[i].bfield_HH(pos)
        return b

    # ---- 支持多组 rz0：可传 [R,Z] 或 [[R1,Z1],[R2,Z2],...] ----
    rz0_arr = np.asarray(rz0, dtype=float)
    if rz0_arr.ndim == 1 and rz0_arr.size == 2:
        seeds = [rz0_arr]
    else:
        seeds = [np.asarray(s, float) for s in rz0_arr]
    # ---------------------------------------------------

    last_err = None
    for seed in seeds:
        def fun(rz):
            try:
                lines = _safe_tracing(field, rz[0], rz[1], phi0, **kwargs)
            except _TraceStalled as e:
                # 打印原因并给大惩罚，优化器会避开；若该 seed 不行，外层会换下一个
                print(f"tracing 警告/卡死, rz = {rz}: {e}")
                return 1e6
            except Exception as e:
                print(f"tracing 失败, rz = {rz}，错误: {e}")
                return 1e6
            r=(lines[0][1][0]-rz[0])**2+(lines[0][1][1]-rz[1])**2
            lines=None
            return r

        try:
            res=minimize(fun, seed, method='L-BFGS-B', bounds=bounds)
        except Exception as e:
            print(f"minimize 异常，seed={seed}，错误: {e}")
            last_err = e
            continue

        print(res)
        if res.success and np.isfinite(res.fun):
            return res.x
        last_err = RuntimeError(f"optimizer failed: {res.message if hasattr(res,'message') else 'unknown'}")

    # 所有 seed 都失败
    if last_err is not None:
        raise last_err
    raise RuntimeError("findax: all seeds failed.")


def findax(coils,rz0=[1,0],phi0=0,bounds=None,**kwargs):
    if bounds is None:
        bounds=[(rz0[0]-0.5,rz0[0]+0.5),(-0.1,0.1)]
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




def plotsurface(s):
    pass



def from_simsopt(simsopt_coils):
    from coilpy import Coil as CoilpyCoil
    """
    Create a coilpy.Coil object from a list of simsopt.Coil objects.
    """
    xx, yy, zz, II, names, groups = [], [], [], [], [], []
    if not isinstance(simsopt_coils, (list, tuple)):
        simsopt_coils = [simsopt_coils]
    for i, sc in enumerate(simsopt_coils):
        gamma = sc.curve.gamma()  # (n_points, 3)
        x, y, z = gamma[:, 0], gamma[:, 1], gamma[:, 2]
        xx.append(list(x))
        yy.append(list(y))
        zz.append(list(z))
        II.append(sc.current.get_value())
        names.append(getattr(sc.curve, 'name', f'coil_{i}'))
        groups.append(1)  # default to group 1, or customize

    return CoilpyCoil(xx=xx, yy=yy, zz=zz, II=II, names=names, groups=groups)



def distance_cp(s, cs): 
    ''' 
    Distance between magnetic surface and coils.

    Args:
        s: magnetic surface object, with method gamma()
        cs: list of simsopt coil objects, each with a curve having method gamma()

    Returns:
        dcp_min (float)         : Minimum distance.
        point_on_coil (ndarray) : Point on coil where distance is minimal.
        point_on_surface (ndarray): Point on magnetic surface where distance is minimal.
    '''
    # 收集所有线圈的坐标点
    coil = []
    for c in cs:
        cur = c.curve
        coil.append(cur.gamma())  # shape: (ns, 3)

    coil = np.array(coil)  # shape: (nc, ns, 3)
    coil = np.ascontiguousarray(coil.reshape(-1, 3), dtype=np.float64)  # flatten to (N1, 3)

    rs = s.gamma()  # shape: (N2, 3)
    rs = np.ascontiguousarray(rs.reshape(-1, 3), dtype=np.float64)

    # 计算每一对点之间的欧几里得距离
    dr = (coil[:, np.newaxis, :] - rs[np.newaxis, :, :])  # shape: (N1, N2, 3)
    dr_norm = np.linalg.norm(dr, axis=-1)  # shape: (N1, N2)

    # 找到最小距离和对应的索引
    min_idx = np.unravel_index(np.argmin(dr_norm), dr_norm.shape)
    idx_coil, idx_surface = min_idx

    dcp_min = dr_norm[idx_coil, idx_surface]
    point_on_coil = coil[idx_coil]
    point_on_surface = rs[idx_surface]

    return dcp_min, point_on_coil, point_on_surface

if __name__ == "__main__":
    from simsopt._core import load, save
    from simsopt.field import coils_to_focus, BiotSavart
    from simsopt.geo import SurfaceRZFourier,CurveXYZFourier,plot
    ID = 400049# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000  
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')
    cpcoils=from_simsopt(coils)
    # poincareplot(cpcoils,rz0=[1,0],len=0.1,show=True)#磁轴位置x: [ 8.763e-01  4.552e-06]


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

    axlist2=tracingFULL(field,[0.95],[0.1],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line2=rzp2curverz(axlist2)

    axlist3=tracingFULL(field,[0.86],[0],niter=1,nstep=360)
    # print(type(axlist))
    # print(axlist)
    line3=rzp2curverz(axlist3)
    from simsopt.geo import plot

    plot([line3,ma])


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





