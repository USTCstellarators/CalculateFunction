import numpy as np
import inspect
from simsopt.geo import SurfaceRZFourier,plotting,SurfaceXYZFourier
from fieldarg import L_grad_B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_cpp_bound(attr):
    try:
        if inspect.isbuiltin(attr):
            return True
        inspect.getsourcefile(attr)
        return False
    except Exception:
        return True

def explore_class(obj_or_cls, max_len=200):
    is_class = isinstance(obj_or_cls, type)
    cls = obj_or_cls if is_class else obj_or_cls.__class__
    obj = None if is_class else obj_or_cls

    print("=" * 60)
    print(f"🔍 Exploring {'class' if is_class else 'instance of'}: {cls.__name__}")
    print(f"📦 Module: {cls.__module__}")
    print("=" * 60)

    print("\n📄 Docstring:\n")
    print(inspect.getdoc(cls) or "No docstring available.")

    print("\n📚 Attributes and Methods:\n")
    names = dir(obj if obj else cls)

    for name in names:
        if name.startswith("_"):
            continue
        try:
            # 对象获取值
            attr_val = getattr(obj if obj else cls, name)
            # 类中获取原始方法（可能未绑定），以便抓取完整签名
            attr_def = getattr(cls, name, attr_val)

            kind = "Method" if callable(attr_val) else "Property"
            src = "C++" if is_cpp_bound(attr_val) else "Python"
            flag = "🧬C++" if src == "C++" else "🐍Py "

            if callable(attr_val):
                try:
                    sig = str(inspect.signature(attr_def))
                except Exception:
                    sig = "(signature unavailable)"
                print(f"{flag} {kind:8} {name:25} [{src}] {sig}")
            else:
                try:
                    val_str = str(attr_val)
                    if len(val_str) > max_len:
                        val_str = val_str[:max_len] + "..."
                    print(f"{flag} {kind:8} {name:25} [{src}] {val_str}")
                except Exception:
                    print(f"{flag} {kind:8} {name:25} [{src}] (access error)")
        except Exception as e:
            print(f"⚠️  Failed to inspect {name}: {e}")

    print("\n✅ Done.\n")



def compare_instances(obj1, obj2, max_len=200, show_all=False, atol=1e-8):
    assert obj1.__class__ == obj2.__class__, "Instances must be of the same class."
    cls = obj1.__class__

    print("=" * 70)
    print(f"🔍 Comparing instances of: {cls.__name__}")
    print("=" * 70)

    names = sorted(set(dir(obj1)).union(dir(obj2)))
    common_names = [n for n in names if not n.startswith("_")]

    for name in common_names:
        try:
            val1 = getattr(obj1, name)
            val2 = getattr(obj2, name)
        except Exception as e:
            print(f"⚠️  Cannot access '{name}': {e}")
            continue

        is_callable = callable(val1)
        src1 = "C++" if is_cpp_bound(val1) else "Python"
        src2 = "C++" if is_cpp_bound(val2) else "Python"
        flag1 = "🧬C++" if src1 == "C++" else "🐍Py "
        flag2 = "🧬C++" if src2 == "C++" else "🐍Py "
        type_str = "Method" if is_callable else "Property"

        if is_callable:
            try:
                sig1 = str(inspect.signature(val1))
            except Exception:  
                sig1 = "(signature unavailable)"
            try:
                sig2 = str(inspect.signature(val2))
            except Exception:
                sig2 = "(signature unavailable)"
            eq = sig1 == sig2 and src1 == src2
            if eq or show_all:
                marker = "✅" if eq else "❌"
                print(f"{marker} {type_str:8} {name}")
                print(f"  {flag1} sig1: {sig1}")
                print(f"  {flag2} sig2: {sig2}")
        else:
            try:
                if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                    equal = np.array_equal(val1, val2)
                    close = np.allclose(val1, val2, atol=atol)
                    diff = not (equal or close)
                    short1 = str(val1)[:max_len] + ("..." if len(str(val1)) > max_len else "")
                    short2 = str(val2)[:max_len] + ("..." if len(str(val2)) > max_len else "")
                elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                    try:
                        equal = val1 == val2
                        diff = not equal
                    except Exception:
                        diff = True
                    short1 = str(val1)[:max_len] + ("..." if len(str(val1)) > max_len else "")
                    short2 = str(val2)[:max_len] + ("..." if len(str(val2)) > max_len else "")
                else:
                    try:
                        equal = val1 == val2
                        diff = not equal
                    except Exception:
                        diff = True
                    short1 = str(val1)[:max_len] + ("..." if len(str(val1)) > max_len else "")
                    short2 = str(val2)[:max_len] + ("..." if len(str(val2)) > max_len else "")
            except Exception as e:
                diff = True
                short1 = "(error)"
                short2 = "(error)"

            if diff or show_all:
                marker = "❌" if diff else "✅"
                print(f"{marker} {type_str:8} {name}")
                print(f"  obj1: {short1}")
                print(f"  obj2: {short2}")

    print("\n✅ Comparison complete.\n")

def nml_to_focus(nml_filename, focus_filename, nfp=2):
    import re
    """
    从 VMEC .nml 文件中提取 RBC/ZBS 的 (n, m) 项，自动计算 bmn,写入 FOCUS 所需的 .boundary 文件。

    参数:
        nml_filename: str, 输入的 .nml 文件路径
        focus_filename: str, 输出的 .boundary 文件路径
        nfp (int): 磁场周期数
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

    return curve



def plot3D(fieldline):
    """
    单纯绘制 fieldline 的 3D 点
    
    参数：
    - fieldline: shape (N,3)，包含 [X, Y, Z]
    """
    X = fieldline[:, 0]
    Y = fieldline[:, 1]
    Z = fieldline[:, 2]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, 'b.', markersize=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fieldline 3D Plot')
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

import numpy as np

def shell_sort(a, n, m):
    k = m // 2
    while k > 0:
        for i in range(k, m):
            j = i - k
            while j >= 0:
                if a[n, j] > a[n, j + k]:
                    temp = np.copy(a[:, j])
                    a[:, j] = a[:, j + k]
                    a[:, j + k] = temp
                    j -= k
                else:
                    break
        k = k // 2
    return a


def rz_resort(R_sorted, Z_sorted, nfp):
    """
    将 RZ 网格映射到 1/(2*nfp) 基本单元，不重新排序
    
    Parameters:
    - R_sorted, Z_sorted: shape (n, m) 原始 RZ 网格，n=toroidal点数，m=poloidal点数
    - nfp: field periods
    
    Returns:
    - R_unit, Z_unit: shape (n_unit, m)
    """
    n, m = R_sorted.shape
    unit = np.pi / nfp  # 基本单元大小
    
    # 生成 phi 对应每行
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # 对偶数单元保持，奇数单元镜像
    flip = ((phi // unit).astype(int) % 2)
    
    R_unit = []
    Z_unit = []
    for i in range(n):
        if phi[i] < unit:  # 只保留第一个基本单元
            R_unit.append(R_sorted[i, :])
            Z_i = Z_sorted[i, :]
            if flip[i] == 1:
                Z_i = -Z_i
            Z_unit.append(Z_i)
    
    R_unit = np.array(R_unit)
    Z_unit = np.array(Z_unit)
    
    return R_unit, Z_unit


def fieldline2rz(fieldline, axisline, m, n, nfp):
    """
    将 fieldline 数据按 poloidal 排序，并返回 (phi, theta, xyz) 网格
    
    Parameters:
    - fieldline: (n*m, 4) array, columns: [x, y, z, φ]
    - axisline: (n+1, 4) array, columns: [x, y, z, φ]
    - m: poloidal resolution (numquadpoints_theta)
    - n: toroidal resolution (numquadpoints_phi)
    - nfp: number of field periods
    
    Returns:
    - xyz: (n, m, 3) array
    """

    # 提取 R 和 Z
    R = np.zeros((n, m))
    Z = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            idx = i + j * n
            x, y, z = fieldline[idx, 0], fieldline[idx, 1], fieldline[idx, 2]
            R[i, j] = np.sqrt(x**2 + y**2)
            Z[i, j] = z

    # 计算 poloidal angle theta
    raxis = np.sqrt(axisline[n // (2 * nfp), 0]**2 + axisline[n // (2 * nfp), 1]**2)
    zaxis = 0.0
    theta = np.zeros(m)
    for j in range(m):
        theta[j] = np.arctan2(Z[n // (2 * nfp), j] - zaxis, R[n // (2 * nfp), j] - raxis)
        if theta[j] < 0:
            theta[j] += 2 * np.pi

    # 排序
    R1 = np.zeros((n+1, m))
    Z1 = np.zeros((n+1, m))
    R1[:n, :] = R
    Z1[:n, :] = Z
    R1[n, :] = theta
    Z1[n, :] = theta
    R_sorted = shell_sort(R1, n, m)[:n, :]
    Z_sorted = shell_sort(Z1, n, m)[:n, :]
    return rz_resort(R_sorted,Z_sorted,nfp)

def fieldline2gamma(fieldline, axisline, m, n, nfp):
    """
    将 fieldline 数据按 poloidal 排序，并返回 (phi, theta, xyz) 网格
    
    Parameters:
    - fieldline: (n*m, 4) array, columns: [x, y, z, φ]
    - axisline: (n+1, 4) array, columns: [x, y, z, φ]
    - m: poloidal resolution (numquadpoints_theta)
    - n: toroidal resolution (numquadpoints_phi)
    - nfp: number of field periods
    
    Returns:
    - xyz: (n, m, 3) array
    """

    # 提取 R 和 Z
    R = np.zeros((n, m))
    Z = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            idx = i + j * n
            x, y, z = fieldline[idx, 0], fieldline[idx, 1], fieldline[idx, 2]
            R[i, j] = np.sqrt(x**2 + y**2)
            Z[i, j] = z

    # 计算 poloidal angle theta
    raxis = np.sqrt(axisline[n // (2 * nfp), 0]**2 + axisline[n // (2 * nfp), 1]**2)
    zaxis = 0.0
    theta = np.zeros(m)
    for j in range(m):
        theta[j] = np.arctan2(Z[n // (2 * nfp), j] - zaxis, R[n // (2 * nfp), j] - raxis)
        if theta[j] < 0:
            theta[j] += 2 * np.pi

    # 排序
    R1 = np.zeros((n+1, m))
    Z1 = np.zeros((n+1, m))
    R1[:n, :] = R
    Z1[:n, :] = Z
    R1[n, :] = theta
    Z1[n, :] = theta
    R_sorted = shell_sort(R1, n, m)[:n, :]
    Z_sorted = shell_sort(Z1, n, m)[:n, :]
    R_sorted,Z_sorted= rz_resort(R_sorted,Z_sorted,nfp)
    phi = np.linspace(0, np.pi/nfp, n // (2 * nfp), endpoint=False)  # 基本单元

    X = R_sorted * np.cos(phi[:, None])
    Y = R_sorted * np.sin(phi[:, None])
    Z = Z_sorted

    XYZ = np.stack([X, Y, Z], axis=-1)  # shape (n, m, 3)
    return XYZ



def rz2surface(R, Z, nfp, mpol, ntor):
    """
    用傅里叶展开拟合 R, Z surface

    R(θ, ζ) = sum_{m,n} R_mn cos(m θ - n N_fp ζ)
    Z(θ, ζ) = sum_{m,n} Z_mn sin(m θ - n N_fp ζ)

    参数：
    - R, Z: shape (n_zeta, n_theta) 对应 (ζ, θ)
    - nfp: 几何周期数
    - mpol: poloidal 最大 m
    - ntor: toroidal 最大 n
    - stellsym: 是否考虑星形对称性
    """
    n_zeta, n_theta = R.shape
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    zeta  = np.linspace(0, np.pi/nfp, n_zeta, endpoint=False)
    Theta, Zeta = np.meshgrid(theta, zeta, indexing='ij')
    
    R_flat = R.T.flatten()  # 转置后 flatten，使第一维是 θ
    Z_flat = Z.T.flatten()
    
    # 1. R 系数
    A_R_list = []
    for m in range(mpol+1):
        if m == 0:
            n_range = range(0, ntor+1)
        else:
            n_range = range(-ntor, ntor+1)
        for n in n_range:
            A_R_list.append(np.cos(m*Theta - n*nfp*Zeta).flatten())
    
    A_R = np.column_stack(A_R_list)
    dofs_R, _, _, _ = np.linalg.lstsq(A_R, R_flat, rcond=1e-12)
    
    # 2. Z 系数
    A_Z_list = []
    for m in range(mpol+1):
        if m == 0:
            n_range = range(1, ntor+1)
        else:
            n_range = range(-ntor, ntor+1)
        for n in n_range:
            A_Z_list.append(np.sin(m*Theta - n*nfp*Zeta).flatten())
    
    A_Z = np.column_stack(A_Z_list)
    dofs_Z, _, _, _ = np.linalg.lstsq(A_Z, Z_flat, rcond=1e-12)
    
    dofs = np.concatenate([dofs_R, dofs_Z])

    surf = SurfaceRZFourier(nfp=nfp, mpol=mpol, ntor=ntor, stellsym=True)
    surf.set_dofs(np.array(dofs))
    print(f"拟合完成: 总 dofs = {len(dofs)}")
    print('dofs =', dofs)
    return surf


def xyz2surface(XYZ, nfp,  mpol, ntor):
    """
    用傅里叶展开拟合 X,Y,Z surface

    X(θ, ζ) = Σ X_mn cos(mθ - nN_fpζ)
    Y(θ, ζ) = Σ Y_mn cos(mθ - nN_fpζ)
    Z(θ, ζ) = Σ Z_mn sin(mθ - nN_fpζ)

    Parameters:
    - XYZ: shape (n, m, 3)，网格点 (ζ, θ, [x,y,z])
    - nfp: 几何周期数
    - mode: [mpol, ntor] 最大展开次数
    """
    n_zeta, n_theta, _ = XYZ.shape

    # 参数角
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    zeta  = np.linspace(0, np.pi/nfp, n_zeta, endpoint=False)
    Theta, Zeta = np.meshgrid(theta, zeta, indexing="ij")

    # 提取 x,y,z 并 flatten
    X_flat = XYZ[:, :, 0].T.flatten()
    Y_flat = XYZ[:, :, 1].T.flatten()
    Z_flat = XYZ[:, :, 2].T.flatten()

    # 1. X 系数
    A_X_list = []
    for m in range(mpol+1):
        if m == 0:
            n_range = range(0, ntor+1)
        else:
            n_range = range(-ntor, ntor+1)
        for n in n_range:
            A_X_list.append(np.cos(m*Theta - n*nfp*Zeta).flatten())
    A_X = np.column_stack(A_X_list)
    dofs_X, _, _, _ = np.linalg.lstsq(A_X, X_flat, rcond=None)

    # 2. Y 系数
    A_Y_list = []
    for m in range(mpol+1):
        if m == 0:
            n_range = range(1, ntor+1)
        else:
            n_range = range(-ntor, ntor+1)
        for n in n_range:
            A_Y_list.append(np.cos(m*Theta - n*nfp*Zeta).flatten())
    A_Y = np.column_stack(A_Y_list)
    dofs_Y, _, _, _ = np.linalg.lstsq(A_Y, Y_flat, rcond=None)

    # 3. Z 系数
    A_Z_list = []
    for m in range(mpol+1):
        if m == 0:
            n_range = range(1, ntor+1)   
        else:
            n_range = range(-ntor, ntor+1)
        for n in n_range:
            A_Z_list.append(np.sin(m*Theta - n*nfp*Zeta).flatten())
    A_Z = np.column_stack(A_Z_list)
    dofs_Z, _, _, _ = np.linalg.lstsq(A_Z, Z_flat, rcond=None)

    # 拼接
    dofs = np.concatenate([dofs_X, dofs_Y, dofs_Z])

    # 构造 surface
    surf = SurfaceXYZFourier(nfp=nfp, mpol=mpol, ntor=ntor)
    surf.set_dofs(dofs)

    print(f"拟合完成: 总 dofs = {len(dofs)}")
    return surf


def fieldline2rzsurface(fieldline, axisline, m, n, nfp,mpol,ntor):
    R,Z=fieldline2rz(fieldline, axisline, m, n, nfp)
    return rz2surface(R,Z,nfp,mpol,ntor)

def fieldline2xyzsurface(fieldline, axisline, m, n, nfp,mpol,ntor):
    return xyz2surface(fieldline2gamma(fieldline, axisline, m, n, nfp),nfp,mpol,ntor)


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


if __name__ == "__main__":

    from simsopt.geo import CurveRZFourier

    # ✅ 支持传入类
    explore_class(CurveRZFourier)

    # # ✅ 支持传入实例
    # curve = CurveRZFourier(quadpoints=100, order=5, nfp=1, stellsym=True)
    # explore_class(curve)
    # # ✅ 也可以直接传入类名
    # print("Exploring CurveRZFourier class directly:")
    # explore_class(CurveRZFourier)

    # from qsc import Qsc
    # from simsopt.geo import SurfaceXYZTensorFourier,SurfaceXYZFourier,SurfaceRZFourier,CurveXYZFourier,CurveRZFourier
    # from simsopt.field import BiotSavart,Coil





    # curve1 = CurveRZFourier(quadpoints=100, order=5, nfp=1, stellsym=True)
    # curve2 = CurveRZFourier(quadpoints=100, order=5, nfp=1, stellsym=True)
    # curve2.set_dofs(curve1.get_dofs() + 0.001)  # Slightly different

    # compare_instances(curve1, curve2)










    # surf=surf.to_RZFourier()
    # #surf.change_resolution(12,12)
    # surf.write_nml('temp.nml')

    # nml_to_focus("temp.nml", "poincare.boundary", nfp=surf.nfp)
