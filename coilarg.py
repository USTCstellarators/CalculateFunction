import numpy as np
def coilkappa(cs):
    '''
    input:simsopt线圈
    '''
    kappalist=[]
    for c in cs:
        cur=c.curve
        kappalist.append(cur.kappa())
    kappalist = np.array(kappalist)
    kappalist=np.abs(kappalist)
    return kappalist

def coiltorsion(cs):
    '''
    input:simsopt线圈
    '''
    torsionlist=[]
    for c in cs:
        cur=c.curve
        torsionlist.append(cur.torsion())  
    torsionlist = np.array(torsionlist)
    torsion=np.abs(torsionlist)
    return torsionlist



def coil_to_para(curve_input, currents, ma=None, nfp=1,stellsym=False,surfaceorder=6,rz0=[0.9,0],max_attempts=1000,max_volume=2.418):
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
            if residual_norm > 1e-10:
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





if __name__ == "__main__":
    import time
    from fieldline import fullax,from_simsopt
    from simsopt._core import load
    from simsopt.field import coils_to_focus, Current, coils_via_symmetries, BiotSavart
    from simsopt.geo import surfacerzfourier,CurveXYZFourier,plot,Volume,SurfaceXYZTensorFourier,create_equally_spaced_curves
    

    ID = 958# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000 
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')

    currents = [c.current.get_value() for c in coils]
    base_curves = [c.curve for c in coils]
    print('len(base_curves)',len(base_curves))
    print('len(currents)',len(currents))

    curve_input=[cur.x for cur in base_curves]
    print('len(curve_input)',len(curve_input))

    start_time = time.time()
    plasma_para,coil_para=coil_to_para(base_curves, currents,nfp=surfaces[0].nfp,stellsym=surfaces[0].stellsym,surfaceorder=6)#
    end_time = time.time()
    print(f"运行时间：{end_time - start_time:.4f} 秒")





    # from simsopt.configs import get_ncsx_data
    # base_curves, base_currents, ma = get_ncsx_data()
    # nfp=3
    # stellsym=True
    # from simsopt.geo import plot
    # items=base_curves.copy()
    # items.append(ma)
    # plot(items,show=True)
    # currents = [current.get_value() for current in base_currents]
    # curve_input=[cur.x for cur in base_curves]
    # print(currents)
    # plasma_para,coil_para=coil_to_para(base_curves, currents, nfp=3,stellsym=True,rz0=[1.589,0],surfaceorder=6)
    
    

