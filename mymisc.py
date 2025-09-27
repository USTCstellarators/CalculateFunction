import numpy as np
import inspect

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




if __name__ == "__main__":

    from simsopt.geo import CurveRZFourier

    # ✅ 支持传入类
    explore_class(CurveRZFourier)

    # ✅ 支持传入实例
    curve = CurveRZFourier(quadpoints=100, order=5, nfp=1, stellsym=True)
    explore_class(curve)
    # ✅ 也可以直接传入类名
    print("Exploring CurveRZFourier class directly:")
    explore_class(CurveRZFourier)

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
