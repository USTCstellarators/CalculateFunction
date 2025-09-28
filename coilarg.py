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

