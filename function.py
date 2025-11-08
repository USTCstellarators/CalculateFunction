import numpy as np
from time import time
from simsopt.geo import (QfmSurface,QfmResidual,boozer_surface_residual,BoozerSurface 
                        , CurveLength, CurveXYZFourier,NonQuasiSymmetricRatio,Volume,plotting,SurfaceXYZTensorFourier, 
                         Iotas,Area,ToroidalFlux)

from simsopt.field import BiotSavart,coils_via_symmetries, Current
import numbers
# from qsc import Qsc
from fieldarg import L_grad_B,distance_cp
from fieldline import fullax,from_simsopt,poincareplot
from simsopt._core.util import Struct
from mymisc import coil2rz0,savefocusinput
import matplotlib.pyplot as plt
from qsc import Qsc

def generate_coils(curve_input, currents, nfp=1, stellsym=True):
    """
    仅支持两种输入：
    1. curve_input 已是 CurveXYZFourier 或其列表；
    2. curve_input.shape = (k, 3, 2*m+1)，currents.shape = (k,)。
    """
    if isinstance(curve_input[0], CurveXYZFourier):
        coils = coils_via_symmetries(curve_input, currents, nfp=nfp, stellsym=stellsym)
    else:
        arr = np.asarray(curve_input)
        k, _, N = arr.shape
        order = (N - 1) // 2
        quad = max(1, 15 * max(order, 1))
        curves = []
        for i in range(k):
            dofs = np.concatenate([arr[i, 0, :], arr[i, 1, :], arr[i, 2, :]])
            c = CurveXYZFourier(quadpoints=quad, order=order)
            c.set_dofs(dofs)
            curves.append(c)
        coils = coils_via_symmetries(curves, [Current(float(c)) for c in currents],
                                     nfp=nfp, stellsym=stellsym)

    print(f'生成 nfp={nfp}, Stellsym={stellsym}, 总共 {len(coils)} 个线圈')
    return coils

def coil_surface_to_para(curve_input,currents,surfaces):
	#生成线圈

    nfp=surfaces[-1].nfp
    stellsym=surfaces[-1].stellsym
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    base_curves=[c.curve for c in coils]
    #biotsavart算磁场,累加电流

    bs = BiotSavart(coils)
    current_sum = sum(i for i in currents)   
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    # 转化成boozer坐标系

    
    qs_error=0
    for s in surfaces:

        print('start')
        volume = Volume(s)
        vol_target = volume.J()

        boozer_surface = BoozerSurface(bs, s, volume, vol_target)

        start_time = time()
        res = boozer_surface.minimize_boozer_exact_constraints_newton(tol=1e-12,G=G0, maxiter=100)
        qs_error+=NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J() 
        end_time = time()
       
        residual_norm = np.linalg.norm(boozer_surface_residual(boozer_surface.surface, res["iota"], res["G"], bs, derivatives=0))
        print(f"vol_target={vol_target:.4f} -> iota={res['iota']:.3f}, volume={volume.J():.3f}, residual={residual_norm:.3e},运行时间：{end_time - start_time:.4f} 秒")
    #可视化
    items=bs.coils.copy()
    items.append(boozer_surface.surface)
    plotting.plot(items,show=True)  
    # # return
    surface=boozer_surface.surface
    qs_error/=len(surfaces)
    boozerresidual=np.linalg.norm(boozer_surface_residual(surface, res['iota'], res['G'], bs, derivatives=0))
    qfm = QfmResidual(surface, bs)
    bnorm=np.linalg.norm(qfm.J())
    iota=res['iota']
    aspect_ratio=surface.aspect_ratio()
    volume=volume.J()

    lgb,_=L_grad_B(coils,surface)
    lgb_min=np.min(lgb)
    curvature= [[c.kappa()] for c in base_curves]
    max_curvature = np.max(curvature)
    torsion= [[c.torsion()] for c in base_curves]
    max_torsion = np.max(torsion)
    length = [CurveLength(c).J() for c in base_curves]
    length=np.array(length)
    total_length=np.sum(length)
    curvesurfacedistance=distance_cp(surface,coils)
    

    plasma_para = Struct()
    plasma_variables = ['surface','qs_error' ,'boozerresidual','bnorm' ,'iota','aspect_ratio','volume','lgb','lgb_min','curvesurfacedistance']
    for v in plasma_variables:
        plasma_para.__setattr__(v, eval(v))

    coil_para = Struct()
    coil_variables = ["coils","curvature","torsion","max_curvature","max_torsion","length","total_length"]
    for v in coil_variables:
        coil_para.__setattr__(v, eval(v))
    # print('-'*20)
    # print("Plasma Parameters:")
    # for key in plasma_para.__dict__:
    #     print(f"{key}: {getattr(plasma_para, key)}")
    # print('-'*20)
    # print("\nCoil Parameters:")
    # for key in coil_para.__dict__:
    #     print(f"{key}: {getattr(coil_para, key)}")
    return plasma_para,coil_para



def coil_to_para(curve_input, currents, ma=None, nfp=1,stellsym=True,surfaceorder=6,rz0=None,max_attempts=1000,max_volume=2.418, residual_tol=1e-10,alpha=0.3):
    '''
    输入k个傅里叶模数为n的线圈, 输出仿星器参数
    curve_input: (k, N) 或者CurveXYZFourier类    
    curve_input: (k, N) 或者CurveXYZFourier类    
    currents: (k,) 
    nfp: int
    stellsym: bool
    alpha: float, alpha越大,步长越大
    '''

    if hasattr(curve_input, '__len__') and hasattr(currents, '__len__'):
        if len(curve_input) != len(currents):
            raise ValueError(f"curve_input 和 currents 长度不一致: len(curve_input) = {len(curve_input)}, len(currents) = {len(currents)}")
    else:
        raise TypeError("curve_input 和 currents 应为可迭代对象（如列表、数组等）")
    
    #生成线圈
    main_start_time = time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)

    print([c.current.get_value() for c in coils])
	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)

    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    if ma is None:
        #用磁场找到磁轴,设定初始面
        cpcoils=from_simsopt(coils)
        ma=fullax(cpcoils,rz0=rz0,niter=1, nstep=10,save=True, save_dir="./opt_steps", prefix="trial")

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
    plotting.plot(items,show=True)

    volume = Volume(surf)#优化变量  
    vol_target =0#目标值
    volumetol=volume.J()#变化单位
    vol_change=volume.J()#每一步变化量

    # 循环
    # 终止条件：达到循环步数；达到目标体积。
    # 每一步增大的体积尝试动态变化
    # 循环
    # 终止条件：达到循环步数；达到目标体积。
    # 每一步增大的体积尝试动态变化

    qs_error=[]
    attempt=1
    while attempt < max_attempts:
        try:
            start_time = time()
            start_time = time()
            s_save = surf.x.copy() #备份
            targetsave=vol_target
            vol_target+=vol_change

            print(f"[{attempt}]  vol_target={vol_target:.4f} ->")
            boozer_surface = BoozerSurface(bs, surf, volume, vol_target)
            res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=100, G=G0)

            final_vol=volume.J()
            residual_norm = np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
            end_time = time()
            end_time = time()
            print(f" iota={res['iota']:.3f}, volume={volume.J():.3f}, residual={residual_norm:.3e},运行时间：{end_time - start_time:.4f} 秒")
            
            # 检查是否收敛
            diff=abs(final_vol - vol_target)/abs(vol_target) 
            if residual_norm > residual_tol or diff > 0.01:
                print(f"第 {attempt} 次发散")
                surf.set_dofs(s_save)
                vol_change=vol_change/2
                vol_target=targetsave
                attempt+=1
                continue

            # 检查是否达到目标
            elif abs(vol_target) > abs(max_volume):
                # surf.set_dofs(s_save)
                print("Volume 达到目标，结束尝试。")
                break

            qs_error.append(NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J())

            volumetol = alpha * vol_change
            volumetol = alpha * vol_change
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

    # boozer_surface = BoozerSurface(bs, surf, volume, max_volume)
    # res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=100, G=G0)
    qs_error.append(NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J())
    residual_norm = np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    main_end_time = time()
    main_end_time = time()
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

    lgb,_=L_grad_B(coils,surface)
    base_curves=[c.curve for c in coils]
    lgb_min=np.min(lgb)    
    curvature= [[c.kappa()] for c in base_curves]
    max_curvature = np.max(curvature)
    torsion= [[c.torsion()] for c in base_curves]
    max_torsion = np.max(torsion)
    length = [CurveLength(c).J() for c in base_curves]
    length=np.array(length)
    total_length=np.sum(length)
    curvesurfacedistance,_,_=distance_cp(surface,coils)

    

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


def coil_to_axis(curve_input, currents, nfp=1,stellsym=True,surfaceorder=6,rz0=None,phi0=0,constraint_weight=1e-3,rtol=1e-6, plot=False,**kwargs):
    '''
    输入k个傅里叶模数为n的线圈, 输出磁轴存在与否, 以及磁轴附近iota和qs_error
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    ''' 


    # 生成线圈
    start_time = time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    [c.fix_all() for c in coils]
    if rz0 is None:
        rz0=[coil2rz0(coils[0]),0]
    print(f'initial guess:rz0={rz0}')
    #用磁场找到磁轴
    # plotting.plot(coils,show=True)

    # cpcoils=from_simsopt(coils)
    ma=fullax(coils,rz0=rz0,phi0=phi0,rtol=rtol,**kwargs)
    # plotting.plot([ma]+coils,show=True)

	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    bstf = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.05, flip_theta=True)

    volume = Volume(surf)
    vol_target=volume.J()
    # vol_target=-0.001
    boozer_surface = BoozerSurface(bs, surf, volume, vol_target)


    # volume = Area(surf)
    # vol_target=volume.J()
    # # vol_target=-0.001
    # boozer_surface = BoozerSurface(bs, surf, volume, vol_target)


    # volume = ToroidalFlux(surf,bstf)
    # vol_target=volume.J()
    # # vol_target=-0.001
    # boozer_surface = BoozerSurface(bs, surf, volume, vol_target)


    
    #####第一步
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=1000,iota=-2,G=G0,constraint_weight=constraint_weight)
    # res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-11, maxiter=1000,iota=-1.9,G=G0,constraint_weight=10)
    end_time1 = time()
    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('boozer1.png') 
        plt.show()
        savefocusinput(surf,None,'boozer1.boundary')
    haveaxis = False
    residual_norm= np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    if residual_norm<1e-9:
        haveaxis = True
    print(f"第一步: iota={res['iota']:.3f}, vol_target={vol_target}, volume={volume.J()}, residual={residual_norm:.3e}, haveaxis={haveaxis}, 运行时间：{end_time1 - start_time:.4f} 秒")
    
    
    #####第二步
    surf2 = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    surf2.set_dofs(surf.get_dofs())



    volume2 = Volume(surf2)
    vol_target2=volume.J()
    # vol_target=-0.001
    boozer_surface2 = BoozerSurface(bs, surf2, volume2, vol_target2)


    # volume2 = Area(surf2)
    # vol_target2=volume.J()
    # # vol_target=-0.001
    # boozer_surface2 = BoozerSurface(bs, surf2, volume2, vol_target2)


    # volume2 = ToroidalFlux(surf2,bstf)
    # vol_target2=volume.J()
    # # vol_target=-0.001
    # boozer_surface2 = BoozerSurface(bs, surf2, volume2, vol_target2)




    boozer_surface2.need_to_run_code = True
    res = boozer_surface2.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=1000,iota=res["iota"],G=res["G"])

    iota=res['iota']
    qs_error=NonQuasiSymmetricRatio(boozer_surface2, BiotSavart(coils)).J()

    residual_norm= np.linalg.norm(boozer_surface_residual(surf2, res["iota"], res["G"], bs, derivatives=0))

    if residual_norm<1e-9:
        haveaxis = True
    
    end_time2 = time()
    
    print(f"第二步: iota={res['iota']:.3f}, vol_target={vol_target2}, volume={volume2.J()}, residual={residual_norm:.3e}, haveaxis={haveaxis}, 运行总时间：{end_time2 - start_time:.4f} 秒")

    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)

        plotting.plot(items,show=False)  
        plt.savefig('boozer2.png') 
        plt.show()
        savefocusinput(surf,None,'boozer2.boundary')
    return haveaxis,iota,qs_error


def coil_to_axis_ar(curve_input, currents, nfp=1,stellsym=True,surfaceorder=6,rz0=None,phi0=0,rtol=1e-6, plot=False,**kwargs):
    '''
    输入k个傅里叶模数为n的线圈, 输出磁轴存在与否, 以及磁轴附近iota和qs_error
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    ''' 


    # 生成线圈
    start_time = time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    [c.fix_all() for c in coils]
    if rz0 is None:
        rz0=[coil2rz0(coils[0]),0]
    print(f'initial guess:rz0={rz0}')
    #用磁场找到磁轴
    # plotting.plot(coils,show=True)

    # cpcoils=from_simsopt(coils)
    ma=fullax(coils,rz0=rz0,phi0=phi0,rtol=rtol,**kwargs)
    # plotting.plot([ma]+coils,show=True)

	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.01, flip_theta=True)
    ar = Area(surf)
    ar_target = ar.J()
    # vol_target=-0.001
    boozer_surface = BoozerSurface(bs, surf, ar, ar_target)
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=1000, iota=0.3,constraint_weight=1000., G=G0)
    end_time1 = time()
    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('boozer1.png')
    haveaxis = False
    residual_norm= np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    if residual_norm<1e-9:
        haveaxis = True
    print(f"第一步: iota={res['iota']:.3f}, area_target={ar_target}, area={ar.J()}, residual={residual_norm:.3e}, haveaxis={haveaxis}, 运行时间：{end_time1 - start_time:.4f} 秒")

    boozer_surface.need_to_run_code = True
    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=1000, G=res['G'])

    iota=res['iota']
    qs_error=NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J()

    residual_norm= np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))

    if residual_norm<1e-9:
        haveaxis = True
    
    end_time2 = time()
    
    print(f"第二步: iota={res['iota']:.3f}, vol_target={ar_target}, volume={ar.J()}, residual={residual_norm:.3e}, haveaxis={haveaxis}, 运行总时间：{end_time2 - start_time:.4f} 秒")

    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('boozer2.png')
    return haveaxis,iota,qs_error


def coil_to_axis_qfm(curve_input, currents, nfp=1,stellsym=True,surfaceorder=8,rz0=None,phi0=0,rtol=1e-6, plot=False,**kwargs):
    '''
    输入k个傅里叶模数为n的线圈, 输出磁轴存在与否, 以及磁轴附近iota和qs_error
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    ''' 


    # 生成线圈
    start_time = time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    [c.fix_all() for c in coils]
    if rz0 is None:
        rz0=[coil2rz0(coils[0]),0]
    print(f'initial guess:rz0={rz0}')

    #用磁场找到磁轴
    # plotting.plot(coils,show=True)

    # cpcoils=from_simsopt(coils)
    ma=fullax(coils,rz0=rz0,phi0=phi0,rtol=rtol,**kwargs)
    # plotting.plot([ma]+coils,show=True)

	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    bs_tf = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.01, flip_theta=True)
    #########第一步

    tf = ToroidalFlux(surf, bs_tf)
    tf_target = tf.J()

    qfm_surface = QfmSurface(bs, surf, tf, tf_target)
    qfm = QfmResidual(surf, bs)
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                            constraint_weight=100)
    end_time1 = time()                                                        
    print(f"第一步qfm, ||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}, 运行时间：{end_time1 - start_time:.4f} 秒")
    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items)  
        # plotting.plot(items,show=False)  
        # plt.savefig('qfm.png')
    #########第二步
    volume = Volume(surf)
    vol_target=volume.J()
    # vol_target=-0.001
    boozer_surface = BoozerSurface(bs, surf, volume, vol_target)
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, iota=-1.9,G=G0,maxiter=1000)

    boozer_surface.need_to_run_code = True

    iota=iota=res['iota']
    haveaxis = False
    residual_norm= np.linalg.norm(boozer_surface_residual(surf, res["iota"], res["G"], bs, derivatives=0))
    # qs_error=NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils)).J()
    if residual_norm<1e-9:
        haveaxis = True
    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items)  
        # plotting.plot(items,show=False)  
        # plt.savefig('boozer.png')
    end_time2 = time()
    
    print(f"第二步: iota={res['iota']:.3f}, vol_target={vol_target}, volume={volume.J()}, residual={residual_norm:.3e}, haveaxis={haveaxis}, 运行总时间：{end_time2 - start_time:.4f} 秒")


    return haveaxis,iota,residual_norm


def coil_to_axis_qfm_both(curve_input, currents, nfp=1,stellsym=True,surfaceorder=8,rz0=None,phi0=0,constraint_weight=1e-4,rtol=1e-8, plot=False,**kwargs):
    '''
    输入k个傅里叶模数为n的线圈, 输出磁轴存在与否, 以及磁轴附近iota和qs_error
    curve_input: (k, N) 或者CurveXYZFourier类
    currents: (k,) 
    nfp: int
    stellsym: bool
    ''' 


    # 生成线圈
    start_time = time()
    coils = generate_coils(curve_input, currents, nfp=nfp, stellsym=stellsym)
    savefocusinput(coils=coils,nfp=nfp)
    if rz0 is None:
        rz0=[coil2rz0(coils[0]),0]
    print(f'initial guess:rz0={rz0}')
    for coil in coils:
        coil.fix_all()  

    #用磁场找到磁轴
    # plotting.plot(coils,show=True)

    # cpcoils=from_simsopt(coils)
    ma=fullax(coils,rz0=rz0,phi0=phi0,rtol=rtol,**kwargs)
    # plotting.plot([ma]+coils,show=True)

	#biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    bs_tf = BiotSavart(coils)
    current_sum = sum(i for i in currents)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.1, flip_theta=True)
    #########第一步

    # tf = ToroidalFlux(surf, bs_tf)
    # tf_target = tf.J()

    tf = Area(surf)
    tf_target = tf.J()

    qfm_surface = QfmSurface(bs, surf, tf, tf_target)
    qfm = QfmResidual(surf, bs)
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-7, maxiter=1000,
                                                            constraint_weight=constraint_weight)
    end_time1 = time()                                                        
    print(f"第一步qfm, ||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}, 运行时间：{end_time1 - start_time:.4f} 秒")

    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('qfm1.png')
        plt.show()
        savefocusinput(surf,None,'qfm1.boundary')

    #########第二步

    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-10, maxiter=1000)




    haveaxis = False

    if np.linalg.norm(qfm.J())<1e-8:
        haveaxis = True
    if plot:
        items=coils.copy()
        items.append(surf)
        items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('qfm2.png')
        plt.show()
        savefocusinput(surf,None,'qfm2.boundary')
    end_time2 = time()

    print(f"第二步qfm, ||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}, 运行总时间：{end_time2 - start_time:.4f} 秒")


    return haveaxis,'qfm',np.linalg.norm(qfm.J())





if __name__ == "__main__":
    #测试coil_to_para

    import numpy as np
    from simsopt.geo import plotting
    from simsopt._core import load, save

    ID = 1448213# ID可在scv文件中找到索引，以958为例

    fID = ID // 1000 
    [surfaces, coils] = load(f'../../projects/QUASR_08072024/simsopt_serials/{fID:04}/serial{ID:07}.json')
    savefocusinput(surface=surfaces[-1],coils=coils,nfp=4)
    print(type(coils[0].current))
    print([c.current.get_value() for c in coils])
    print(len(coils))


    nfp=surfaces[0].nfp
    num=int(len(coils)/2/nfp)
    
    from fieldline import fullax

    # # ma=fullax(coils,rz0=[0.79,0],phi0=0,rtol=1e-8)
    # # print(ma.rc)
    # # print(ma.zs)
    # # rc=[9.99057567e-01, -6.48764712e-05, -6.51414492e-05, -6.49495070e-05, -2.54225515e-01, -6.50062773e-05, -6.49079477e-05, -6.50085358e-05,  5.97650514e-02, -6.49762142e-05 ,-6.50107840e-05]
    # # zs=[0,-6.73979147e-08,  1.16730845e-07,  1.51527227e-08,  2.48975999e-01,  3.85058886e-08, -5.93257535e-08,  3.35021954e-08, -5.95129484e-02 , 1.43755548e-09,  3.39886970e-08]
    # rc=[9.99057567e-01, -6.48764712e-05, -6.51414492e-05, -6.49495070e-05]
    # zs=[0,-6.73979147e-08,  1.16730845e-07,  1.51527227e-08]
    # # rc=[9.99057567e-01, -6.48764712e-05, -6.51414492e-05]
    # # zs=[0,-6.73979147e-08,  1.16730845e-07]
    # for etabar in [0.5]:
    #     stel = Qsc(rc=rc, zs=zs, nfp=surfaces[0].nfp, etabar=etabar)
    #     # stel = Qsc(rc=ma.rc, zs=ma.zs, nfp=2, etabar=etabar)
    #     print(f"etabar={etabar}: iota={stel.iota}, lgb={stel.min_L_grad_B}")
    #     stel.plot_boundary() # Plot the flux surface shape at the default radius r=1
    #     plt.savefig(f'simple_demo_etabar={etabar}.png', dpi=300)


    # haveaxis,iota,qs_error=coil_to_axis_qfm_both([c.curve for c in coils[0:num]],[c.current.get_value() for c in coils[0:num]],nfp=nfp,constraint_weight=1e3,rtol=1e-8,phi0=0,method='BDF',plot=True)





    # print(haveaxis)
    # print(iota)
    # print(qs_error)

    










    nfp=surfaces[0].nfp
    constraint_weight=1e4
    rtol=1e-8
    method='BDF'
    plot=True

    surfaceorder=6

    print(len(coils))


    ma=fullax(coils,phi0=0,rtol=rtol)
    plotting.plot([ma]+coils,show=True)

	# #biotsavart算磁场,累加电流
    bs = BiotSavart(coils)
    bs_tf = BiotSavart(coils)

    mpol = surfaceorder  
    ntor = surfaceorder  
    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    surf = SurfaceXYZTensorFourier(
    mpol=surfaceorder, ntor=surfaceorder, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)#
    # surf.least_squares_fit(surfRZ.gamma())
    surf.fit_to_curve(ma, 0.1, flip_theta=True)
    #########第一步


    # surf=surfaces[3]
    tf = ToroidalFlux(surf, bs_tf)
    tf_target = tf.J()

    # tf = Area(surf)
    # tf_target = tf.J()

    qfm_surface = QfmSurface(bs, surf, tf, tf_target)
    qfm = QfmResidual(surf, bs)
    start_time = time()
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-33, maxiter=10000,
                                                            constraint_weight=constraint_weight)
    end_time1 = time()                                                        
    print(f"第一步qfm, ||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}, 运行时间：{end_time1 - start_time:.4f} 秒")

    if plot:
        items=coils.copy()
        items.append(surf)
        # items.append(ma)
        plotting.plot(items,show=False)  
        plt.savefig('qfm1.png')
        plt.show()
        savefocusinput(surf,None,'qfm1.boundary')


