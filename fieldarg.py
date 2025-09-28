import numpy as np
from simsopt.field import BiotSavart
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Bmod(s,cs,po=None):
    '''
    input:
        s,simsopt面
        cs,simsopt线圈
        po,计算网格
    '''
    field = BiotSavart(cs)  # Multiple coils can be included in the list
    if po is None:
        po=s.gamma()
        print(f"po.shape={po.shape}")
    (m,n,_)=po.shape
    po = np.ascontiguousarray(po.reshape(-1, 3), dtype=np.float64)

    field.set_points(np.array(po))

    modB = field.AbsB()

    modB=modB.reshape(m, n)
    return modB


def L_grad_B(cs,s=None,po=None):
    '''
    input:
        s,simsopt面
        cs,simsopt线圈
        po,计算网格
    '''
    from simsopt.field import BiotSavart
    field = BiotSavart(cs)  # Multiple coils can be included in the list
    if po is None:
        if s is None:
            raise ValueError("Either surface s or points po must be provided.")
        po=s.gamma()
    (m,n,_)=po.shape
    po = np.ascontiguousarray(po.reshape(-1, 3), dtype=np.float64)

    field.set_points(np.array(po))

    modB = field.AbsB()

    grad_B__XX , grad_B__XY , grad_B__XZ = field.dB_by_dX()[:,0,0],field.dB_by_dX()[:,1,0],field.dB_by_dX()[:,2,0]

    grad_B__YX ,grad_B__YY ,grad_B__YZ = field.dB_by_dX()[:,0,1],field.dB_by_dX()[:,1,1],field.dB_by_dX()[:,2,1]

    grad_B__ZX ,grad_B__ZY ,grad_B__ZZ = field.dB_by_dX()[:,0,2],field.dB_by_dX()[:,1,2],field.dB_by_dX()[:,2,2]

    grad_B_double_dot_grad_B = (
        grad_B__XX * grad_B__XX
        + grad_B__XY * grad_B__XY
        + grad_B__XZ * grad_B__XZ
        + grad_B__YX * grad_B__YX
        + grad_B__YY * grad_B__YY
        + grad_B__YZ * grad_B__YZ
        + grad_B__ZX * grad_B__ZX
        + grad_B__ZY * grad_B__ZY
        + grad_B__ZZ * grad_B__ZZ
    )

    L_grad_B = modB * np.sqrt(2 / grad_B_double_dot_grad_B)
    L_grad_B=L_grad_B[0].reshape(m, n)
    #print(L_grad_B)
    # 找最小值和对应坐标
    min_idx = np.argmin(L_grad_B)
    min_point = po[min_idx]
    return L_grad_B,min_point

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
