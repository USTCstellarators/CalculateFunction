def generate_coils(curve_input, currents, nfp=1, stellsym=False):
    """
    根据 curve_input 与 currents 生成带对称性的 coils 对象。
    
    参数
    ----
    curve_input : (k,N)、(k,3,N) numpy.ndarray 或 CurveXYZFourier/list
        - (k,N): k 根线圈的 DOF，每根线圈已按 simsopt 的 dofs 顺序拼好
        - (k,3,N): k 根线圈，3 表示 x/y/z，最后一维按 [c0, s1, c1, ...] 排列
        - CurveXYZFourier 或 其列表：直接使用
    currents : array-like, shape (k,)
        每根线圈的电流
    nfp : int
        poloidal field periods
    stellsym : bool
        是否施加 stellarator 对称
    
    返回
    ----
    coils : simsopt.field.BiotSavart
        可直接用于 Biot–Savart 计算的 coils 对象
    """
    import numpy as np
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import coils_via_symmetries, Current

    # ---- 兼容各种输入 ----
    def _to_curves(curve_input, currents):
        """内部：把输入统一成 [CurveXYZFourier], [Current], k, order"""
        if isinstance(curve_input, CurveXYZFourier):
            base_curves = [curve_input]
            order = curve_input.order
            return base_curves, [Current(float(currents[0] if np.ndim(currents) else currents))], 1, order

        if isinstance(curve_input, (list, tuple)) and len(curve_input) and isinstance(curve_input[0], CurveXYZFourier):
            order = curve_input[0].order
            return list(curve_input), [Current(float(c)) for c in currents], len(curve_input), order

        arr = np.asarray(curve_input)
        if arr.ndim == 2:      # (k, N)
            k, N = arr.shape
            order = int((N - 3) // 6) if N > 3 else 0
            quad = max(1, 15 * order)
            curves = []
            for i in range(k):
                c = CurveXYZFourier(quadpoints=quad, order=order)
                c.set_dofs(arr[i])
                curves.append(c)
            return curves, [Current(float(c)) for c in currents], k, order

        if arr.ndim == 3:      # (k, 3, 2*m+1)
            k, xyz, N = arr.shape
            if xyz != 3:
                raise ValueError("curve_input shape must be (k,3,N)")
            order = (N - 1) // 2
            quad = max(1, 15 * order)
            curves = []
            for i in range(k):
                dofs = np.concatenate([arr[i, 0, :], arr[i, 1, :], arr[i, 2, :]])
                c = CurveXYZFourier(quadpoints=quad, order=order)
                c.set_dofs(dofs)
                curves.append(c)
            return curves, [Current(float(c)) for c in currents], k, order

        raise TypeError("curve_input must be CurveXYZFourier, (k,N) or (k,3,N) array")

    base_curves, base_currents, k, coilorder = _to_curves(curve_input, currents)
    print(f"generate_coils: {k} coils, order={coilorder}")
    print("currents:", [c for c in currents])

    # ---- 生成带对称性的 coils ----
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
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
    from fieldline import fullax,from_simsopt,distance_cp,poincareplot
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

    ID = 958# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000 
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')

    currents = [c.current.get_value() for c in coils]
    order=4
    print(surfaces[0].nfp)
    base_curves = [c.curve for c in coils]
    curve_input=[cur.x for cur in base_curves]
    print(curve_input)
    print(np.array(curve_input).shape)

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




