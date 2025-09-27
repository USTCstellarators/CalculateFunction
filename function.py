def generate_coils(curve_input, currents, nfp=1, stellsym=False):
    """
    根据 curve_input 与 currents 生成带对称性的 coils 对象。
    支持两种用法：
      A) curve_input = 基线圈列表(k_base)，currents = k_base
      B) curve_input = 已展开后的全部线圈(k_full)，currents = k_base
         若满足 k_full == k_base * nfp * (2 if stellsym else 1)，自动复制电流

    返回：coils（可用于 Biot–Savart）
    """
    import numpy as np
    import numbers
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import coils_via_symmetries, Current as SOCurrent

    def _as_curve_list(curve_input):
        # 统一出曲线列表，以及每根曲线的 order
        if isinstance(curve_input, CurveXYZFourier):
            return [curve_input], curve_input.order
        if isinstance(curve_input, (list, tuple)) and len(curve_input) and isinstance(curve_input[0], CurveXYZFourier):
            return list(curve_input), curve_input[0].order
        arr = np.asarray(curve_input)
        if arr.ndim == 2:  # (k, N) dofs
            k, N = arr.shape
            order = int((N - 3) // 6) if N > 3 else 0
            quad = max(1, 15 * max(order, 1))
            curves = []
            for i in range(k):
                c = CurveXYZFourier(quadpoints=quad, order=order)
                c.set_dofs(arr[i])
                curves.append(c)
            return curves, order
        if arr.ndim == 3:  # (k, 3, 2*m+1)
            k, xyz, N = arr.shape
            if xyz != 3:
                raise ValueError("curve_input shape 必须为 (k,3,N)")
            order = (N - 1) // 2
            quad = max(1, 15 * max(order, 1))
            curves = []
            for i in range(k):
                dofs = np.concatenate([arr[i, 0, :], arr[i, 1, :], arr[i, 2, :]])
                c = CurveXYZFourier(quadpoints=quad, order=order)
                c.set_dofs(dofs)
                curves.append(c)
            return curves, order
        raise TypeError("curve_input 必须是 CurveXYZFourier、(k,N) 或 (k,3,N) 数组")

    def _to_current_obj_list(vals):
        """把电流元素统一成 simsopt 的 Current/ScaledCurrent/兼容对象列表（保留已是对象的情况）"""
        arr = np.asarray(vals, dtype=object).ravel()
        out = []
        for c in arr:
            if isinstance(c, numbers.Number) or (np.isscalar(c) and not isinstance(c, (str, bytes))):
                out.append(SOCurrent(float(c)))
                continue
            has_get = hasattr(c, "get_value") and callable(getattr(c, "get_value"))
            has_vjp = hasattr(c, "vjp") and callable(getattr(c, "vjp"))
            if has_get and has_vjp:
                out.append(c)
                continue
            try:
                out.append(SOCurrent(float(c)))
            except Exception as e:
                raise TypeError(
                    "currents 中的元素既不是数值，也不是 Current/ScaledCurrent 兼容对象，"
                    f"并且无法转换为 float：{type(c)}"
                ) from e
        return out

    # ---- 曲线表与数量 ----
    base_curves, order = _as_curve_list(curve_input)
    k = len(base_curves)

    # ---- 处理电流数量匹配 ----
    curr_in = np.asarray(currents, dtype=object).ravel()
    m = curr_in.size  # 提供的电流个数（可能是基线圈数）

    sym_factor = int(nfp) * (2 if stellsym else 1)

    if m == k:
        # 情况1：电流数量与曲线数一致
        curr_list = _to_current_obj_list(curr_in)
    elif sym_factor > 0 and (k % sym_factor == 0) and (m == k // sym_factor):
        # 情况2：电流数量是“基线圈数”，而曲线是已经展开到全部线圈
        # 这里按对称因子复制：每个基线圈的电流 → 复制 nfp*(stellsym?2:1) 次
        curr_base = _to_current_obj_list(curr_in)
        # 注意：若 curr_base 里有 Current/ScaledCurrent 对象，复制引用即可（和曲线实例一一对应）
        curr_list = list(np.tile(curr_base, sym_factor))
    else:
        # 情况3：不匹配，明确报错，提示如何修正
        raise ValueError(
            f"currents 长度={m} 与曲线数 k={k} 不匹配；"
            f"若 currents 是基线圈电流，请确保 k == m * nfp * (2 if stellsym else 1)；"
            f"当前 nfp={nfp}, stellsym={stellsym} → 期望 k == {m} * {sym_factor} = {m*sym_factor}。"
        )

    print(f"coils number: {k}, order={order}, nfp={nfp}, stellsym={stellsym}, sym_factor={sym_factor}")
    # 仅尝试打印 get_value()；失败则打印对象
    try:
        print("currents (values):", [getattr(c, "get_value", lambda: c)() if hasattr(c, "get_value") else c for c in curr_list])
    except Exception:
        print("currents (objects):", curr_list)

    # 若 curve_input 本身是“基线圈”，推荐把“基线圈+电流”交给 simsopt 去做对称展开：
    #   coils = coils_via_symmetries(base_curves, curr_base, nfp, stellsym)
    # 但此处你的 curve_input 已经是“全部线圈”时，我们直接把它们和 curr_list 传进去也是可以的。
    # coils_via_symmetries 会再做一次展开；所以为了不重复展开，这里应传“基线圈+基电流”。
    # 判断是否为“已展开”的输入：k == m * sym_factor（当我们刚刚复制过）
    from simsopt.field import coils_via_symmetries
    if m == k:
        # 看起来 curve_input 和 currents 已经对应全部线圈；此时不应再做对称展开
        # 但 coils_via_symmetries 的 API 需要“基线圈+基电流”。为了保持一致，这里采用简单策略：
        # 当 m==k（已一一对应全部线圈）时，构造 nfp=1, stellsym=False 的“无展开”调用：
        coils = coils_via_symmetries(base_curves, curr_list, 1, False)
    else:
        # 我们识别到 currents 是“基线圈电流”（刚刚复制出的 curr_list 是“全部线圈电流”）
        # 为了避免双重展开，应该把“基线圈+基电流”传给 coils_via_symmetries，再由它展开
        k_base = m
        base_curves_for_expand = base_curves[:k_base] if (k == k_base * sym_factor) else base_curves
        curr_base = _to_current_obj_list(curr_in)
        coils = coils_via_symmetries(base_curves_for_expand, curr_base, nfp, stellsym)

    return coils


def coil_surface_to_para(surfaces,curve_input,currents):
    import numpy as np
    from time import time
    from fieldarg import L_grad_B,CurveSurfaceDistance
    from simsopt.geo import (QfmResidual,CurveXYZFourier,boozer_surface_residual,BoozerSurface 
                            , CurveLength, NonQuasiSymmetricRatio, Iotas,Volume)
    from simsopt.field import BiotSavart, coils_via_symmetries,Current
	#生成线圈
    if not isinstance(curve_input[0], (CurveXYZFourier)):
        print("curve_input is not CurveXYZFourier, converting to CurveXYZFourier")
        curve_input=np.array(curve_input)
        (k,N)=curve_input.shape
        coilorder=int((N-3)/6)
        numquadpoints = 15 * coilorder
        base_curves = [CurveXYZFourier(quadpoints=numquadpoints, order=coilorder) for _ in range(k)]
        for i in range(k):
            base_curves[i].x= curve_input[i]
    else:
        base_curves = curve_input
        k = len(base_curves)
        coilorder = base_curves[0].order  

    nfp=surfaces[-1].nfp
    stellsym=surfaces[-1].stellsym
    base_currents= [Current(c) for c in currents]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    current_sum = sum(i for i in currents)   
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    # 转化成boozer坐标系

    
    qs_error=0
    for s in surfaces:

        
        volume = Volume(s)
        vol_target = volume.J()

        boozer_surface = BoozerSurface(bs, s, volume, vol_target)

        start_time = time.time()
        res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=100)
        qs_error+=NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J() 
        end_time = time.time()
       
        residual_norm = np.linalg.norm(boozer_surface_residual(boozer_surface.surface, res["iota"], res["G"], bs, derivatives=0))
        print(f"vol_target={vol_target:.4f} -> iota={res['iota']:.3f}, volume={volume.J():.3f}, residual={residual_norm:.3e},运行时间：{end_time - start_time:.4f} 秒")
    #可视化
    items=bs.coils.copy()
    items.append(boozer_surface.surface)
    plot(items,show=True)  
    # # return
    surface=boozer_surface.surface
    qs_error/=len(surfaces)
    boozerresidual=np.linalg.norm(boozer_surface_residual(surface, res['iota'], res['G'], bs, derivatives=0))
    qfm = QfmResidual(surface, bs)
    bnorm=np.linalg.norm(qfm.J())
    iota=res['iota']
    aspect_ratio=surface.aspect_ratio()
    volume=volume.J()

    lgb,_=L_grad_B(surface,coils)
    lgb_min=np.min(lgb)
    curvature= [[c.kappa()] for c in base_curves]
    max_curvature = np.max(curvature)
    torsion= [[c.torsion()] for c in base_curves]
    max_torsion = np.max(torsion)
    length = [CurveLength(c).J() for c in base_curves]
    length=np.array(length)
    total_length=np.sum(length)
    curvesurfacedistance=CurveSurfaceDistance(surface,coils)
    

    from simsopt._core.util import Struct
    plasma_para = Struct()
    plasma_variables = ['surface','qs_error' ,'boozerresidual','bnorm' ,'iota','aspect_ratio','volume','lgb','lgb_min','curvesurfacedistance']
    for v in plasma_variables:
        plasma_para.__setattr__(v, eval(v))

    coil_para = Struct()
    coil_variables = ["coils","curvature","torsion","max_curvature","max_torsion","length","total_length"]
    for v in coil_variables:
        coil_para.__setattr__(v, eval(v))
    print('-'*20)
    print("Plasma Parameters:")
    for key in plasma_para.__dict__:
        print(f"{key}: {getattr(plasma_para, key)}")
    print('-'*20)
    print("\nCoil Parameters:")
    for key in coil_para.__dict__:
        print(f"{key}: {getattr(coil_para, key)}")
    return plasma_para,coil_para



def coil_to_para(curve_input, currents, ma=None, nfp=1,stellsym=False,surfaceorder=6,rz0=None,max_attempts=1000,max_volume=2.418, residual_tol=1e-10):
    '''
    输入k个傅里叶模数为n的线圈, 输出仿星器参数
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    '''
    import time
    import numpy as np
    from qsc import Qsc
    from fieldline import fullax,from_simsopt,distance_cp,poincareplot
    from fieldarg import L_grad_B,CurveSurfaceDistance
    from simsopt.geo import (SurfaceRZFourier,plot,QfmResidual,CurveXYZFourier,boozer_surface_residual,SurfaceXYZTensorFourier, 
                            BoozerSurface,MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas,Volume,Area)
    from simsopt.field import BiotSavart, coils_via_symmetries,Current

	
    if hasattr(curve_input, '__len__') and hasattr(currents, '__len__'):
        if len(curve_input) != len(currents):
            raise ValueError(f"curve_input 和 currents 长度不一致: len(curve_input) = {len(curve_input)}, len(currents) = {len(currents)}")
    else:
        raise TypeError("curve_input 和 currents 应为可迭代对象（如列表、数组等）")
    
    #生成线圈
    main_start_time = time.time()
    if not isinstance(curve_input[0], (CurveXYZFourier)):
        print("curve_input is not CurveXYZFourier, converting to CurveXYZFourier")
        curve_input=np.array(curve_input)
        (k,N)=curve_input.shape
        print(k)
        coilorder=int((N-3)/6)
        numquadpoints = 15 * coilorder
        base_curves = [CurveXYZFourier(quadpoints=numquadpoints, order=coilorder) for _ in range(k)]
        for i in range(k):
            base_curves[i].x= curve_input[i]
            base_curves[i].set_dofs(base_curves[i].x)
        base_currents= [Current(c) for c in currents]

    else:
        print("curve_input is CurveXYZFourier, using it directly")
        base_curves = curve_input.copy()
        k = len(base_curves)
        coilorder = base_curves[0].order
        base_currents= [Current(c) for c in currents]

    print([c for c in currents])
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    print([c.current.get_value() for c in coils])
	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    if ma is None:
        #用磁场找到磁轴,设定初始面
        cpcoils=from_simsopt(coils)
        ma=fullax(cpcoils,rz0=rz0)

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.02, flip_theta=True)
    items=coils.copy()
    items.append(surf)
    items.append(ma)
    plot(items,show=True)

    volume = Volume(surf)#优化变量  
    vol_target =0#目标值
    volumetol=volume.J()#变化单位
    vol_change=volume.J()#每一步变化量

# 循环
# 终止条件：达到循环步数；达到目标体积。
# 每一步增大的体积尝试动态变化

    qs_error=[]
    attempt=1
    while attempt < max_attempts:
        try:
            start_time = time.time()
            s_save = surf.x.copy() #备份
            targetsave=vol_target
            vol_target+=vol_change

            print(f"[{attempt}]  vol_target={vol_target:.4f} ->")
            boozer_surface = BoozerSurface(bs, surf, volume, vol_target)
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=100, G=G0)

            final_vol=volume.J()
            residual_norm = np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
            end_time = time.time()
            print(f" iota={res['iota']:.3f}, volume={volume.J():.3f}, residual={residual_norm:.3e},运行时间：{end_time - start_time:.4f} 秒")
            
            # 检查是否收敛
            if residual_norm > residual_tol:
                print(f"第 {attempt} 次发散")
                surf.set_dofs(s_save)
                vol_change=vol_change/2
                vol_target=targetsave
                attempt+=1
                continue

            # 检查是否达到目标
            elif abs(final_vol) > abs(max_volume):
                qs_error.append(NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J())
                print("Volume 达到目标，结束尝试。")
                break

            qs_error.append(NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J())

            volumetol = 0.3 * vol_change
            print('volumetol=',volumetol)
            print('vol_change=',vol_change)
            vol_change+=volumetol
            attempt+=1

        except Exception as e:
            print(f"第 {attempt} 次失败, 错误信息: {e}")
            surf.set_dofs(s_save)
            vol_change=vol_change/2
            vol_target=targetsave
            attempt+=1
            continue


    residual_norm = np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    main_end_time = time.time()
    print(f"[最终结果] vol_target={vol_target:.4f} -> iota={res['iota']:.3f}, volume={volume.J():.3f}, residual={residual_norm:.3e}, 运行总时间：{main_end_time - main_start_time:.4f} 秒")


    # return
    surface=surf
    qs_error=np.sum(qs_error)/len(qs_error)
    boozerresidual=np.linalg.norm(boozer_surface_residual(surface, res['iota'], res['G'], bs, derivatives=0))
    qfm = QfmResidual(surface, bs)
    bnorm=np.linalg.norm(qfm.J())
    iota=res['iota']
    aspect_ratio=surface.aspect_ratio()
    volume=volume.J()

    lgb,_=L_grad_B(surface,coils)
    lgb_min=np.min(lgb)    
    curvature= [[c.kappa()] for c in base_curves]
    max_curvature = np.max(curvature)
    torsion= [[c.torsion()] for c in base_curves]
    max_torsion = np.max(torsion)
    length = [CurveLength(c).J() for c in base_curves]
    length=np.array(length)
    total_length=np.sum(length)
    curvesurfacedistance,_,_=CurveSurfaceDistance(surface,coils)
    

    from simsopt._core.util import Struct
    plasma_para = Struct()
    plasma_variables = ['surface','qs_error' ,'boozerresidual','bnorm' ,'iota','aspect_ratio','volume','lgb','lgb_min','curvesurfacedistance']
    for v in plasma_variables:
        plasma_para.__setattr__(v, eval(v))

    coil_para = Struct()
    coil_variables = ["coils","curvature","torsion","max_curvature","max_torsion","length","total_length"]
    for v in coil_variables:
        coil_para.__setattr__(v, eval(v))
    print('-'*20)
    print("Plasma Parameters:")
    for key in plasma_para.__dict__:
        print(f"{key}: {getattr(plasma_para, key)}")
    print('-'*20)
    print("\nCoil Parameters:")
    for key in coil_para.__dict__:
        print(f"{key}: {getattr(coil_para, key)}")
    return plasma_para,coil_para


def coil_to_axis(curve_input, currents, nfp=1,stellsym=False,surfaceorder=6,rz0=None, plot=False):
    '''
    输入k个傅里叶模数为n的线圈, 输出磁轴存在与否, 以及磁轴附近iota和qs_error
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    ''' 
    import time
    import numpy as np
    from qsc import Qsc
    from fieldline import fullax,from_simsopt,distance_cp,poincareplot,fullax_safe
    from fieldarg import L_grad_B,CurveSurfaceDistance
    from simsopt.geo import (SurfaceRZFourier,plot,QfmResidual,CurveXYZFourier,boozer_surface_residual,SurfaceXYZTensorFourier, 
                            BoozerSurface,MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas,Volume,Area)
    from simsopt.field import BiotSavart, coils_via_symmetries,Current
    # 如果 generate_coils 不在同一文件，请在此处导入：
    # from your_module import generate_coils

    surfaceorder=4

    # 生成线圈（仅改动这里）
    start_time = time.time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    print([c for c in currents])

	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))


    #用磁场找到磁轴
    cpcoils=from_simsopt(coils)
    ma=fullax(cpcoils,rz0=rz0)

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.005, flip_theta=True)
    volume = Volume(surf)
    # vol_target=volume.J()
    vol_target=-0.0001
    boozer_surface = BoozerSurface(bs, surf, volume, vol_target)
    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=100, G=G0)

    iota=res['iota']
    qs_error=NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J()

    haveaxis = False
    residual_norm= np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    if residual_norm<1e-9:
        haveaxis = True
    end_time = time.time()
    
    print(f" iota={res['iota']:.3f}, vol_target={vol_target}, volume={volume.J()}, residual={residual_norm:.3e},运行时间：{end_time - start_time:.4f} 秒")

    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plot(items,show=True)
    return haveaxis,iota,qs_error





if __name__ == "__main__":
    # #测试coil_to_para
    # import numpy as np
    # from simsopt.geo import plot
    # from simsopt._core import load, save

    # ID = 958# ID可在scv文件中找到索引，以958为例
    # fID = ID // 1000 
    # [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')

    # currents = [c.current.get_value() for c in coils]
    # order=4
    # print(surfaces[0].nfp)
    # base_curves = [c.curve for c in coils]
    # curve_input=[cur.x for cur in base_curves]
    # print(curve_input)
    # print(np.array(curve_input).shape)

    # plasma_para,coil_para=coil_to_para(base_curves, currents,nfp=surfaces[0].nfp,stellsym=True,surfaceorder=6)

    #测试coil_to_axis
    import numpy as np
    from simsopt.geo import plot
    from simsopt._core import load, save

    ID = 958# ID可在scv文件中找到索引，以958为例 2408903
    fID = ID // 1000 
    [surfaces, coils] = load(f'../simsopt_serials/{fID:04}/serial{ID:07}.json')

    currents = [c.current.get_value() for c in coils]
    order=4
    base_curves = [c.curve for c in coils]
    curve_input=[cur.x for cur in base_curves]
    print('nfp',surfaces[0].nfp)
    # print(curve_input)
    # print(np.array(curve_input).shape)





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



    surf=surfaces[-1]

    surf=surf.to_RZFourier()
    #surf.change_resolution(12,12)
    surf.write_nml('temp.nml')




    from coilpy.surface import FourSurf




    nml_to_focus("temp.nml", "poincare.boundary", nfp=surf.nfp)



    def coil_to_curves_currents(coils):
        from simsopt.geo import CurveXYZFourier
        curves = []
        currents = []
        for c in coils:
            if isinstance(c.curve, CurveXYZFourier):
                curves.append(c.curve)
                currents.append(c.current)
            else:
                pass
                #curves.append(c.curve.curve)
        return curves,currents
    [curves,currents] = coil_to_curves_currents(coils)

    print(surfaces[0].nfp)
    from simsopt.field import coils_to_focus
    coils_to_focus('poincare.focus',curves, currents, nfp=surfaces[0].nfp,stellsym=True)



















    haveaxis,iota,qs_error=coil_to_axis(base_curves, currents,nfp=surfaces[0].nfp,stellsym=True,surfaceorder=6, plot=True)

    print(haveaxis)
    print(iota)
    print(qs_error)



    # surf=surfaces[-1]

    # surf=surf.to_RZFourier()
    # #surf.change_resolution(12,12)
    # surf.write_nml('temp.nml')
    # from coilpy.surface import FourSurf
    # from fieldline import nml_to_focus
    # from simsopt.field import coils_to_focus
    # from simsopt.geo import surfacerzfourier,CurveXYZFourier,plot
    # nml_to_focus("temp.nml", "958.boundary", nfp=surf.nfp)
    # def coil_to_curves_currents(coils):
    #     curves = []
    #     curents = []
    #     for c in coils:
    #         if isinstance(c.curve, CurveXYZFourier):
    #             curves.append(c.curve)
    #             curents.append(c.current)
    #         else:
    #             pass
    #             #curves.append(c.curve.curve)
    #     return curves,curents
    # [curves,currents] = coil_to_curves_currents(coils)
    # coils_to_focus('full_step1.focus',curves, currents, nfp=2,stellsym=True, Ifree=True, Lfree=True)




