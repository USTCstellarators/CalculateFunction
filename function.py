

def coil_surface_to_para(surfaces,curve_input,currents):
    import numpy as np
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



if __name__ == "__main__":
    from simsopt.geo import plot
    from simsopt._core import load
    import time
    ID = 958# ID可在scv文件中找到索引，以958为例
    fID = ID // 1000 
    [surfaces, coils] = load(f'./inputs/serial{ID:07}.json')

    currents = [c.current.get_value() for c in coils]
    base_curves = [c.curve for c in coils]
    start_time = time.time() 
    plasma_para,coil_para=coil_surface_to_para(surfaces,base_curves, currents)
    end_time = time.time()
    print(f"       运行时间：{end_time - start_time:.4f} 秒")

    print('qs_error=',plasma_para.qs_error)



