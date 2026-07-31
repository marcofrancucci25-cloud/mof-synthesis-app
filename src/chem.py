import re, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METALS=json.loads((ROOT/'models/metal_properties.json').read_text())

COMMON_LIGAND_ALIASES={
 'h2bdc':'terephthalic acid','bdc':'terephthalate / terephthalic acid','btc':'benzene-1,3,5-tricarboxylic acid',
 'h3btc':'benzene-1,3,5-tricarboxylic acid','bpdc':'biphenyl-4,4-dicarboxylic acid','dobdc':'2,5-dihydroxyterephthalic acid',
 'bpy':'2,2-bipyridine','4,4-bpy':'4,4-bipyridine','bpz':'4,4-bipyrazole','h2bpz':'4,4-bipyrazole',
 'imidazole':'imidazole','2-methylimidazole':'2-methylimidazole','hmim':'2-methylimidazole','trimesic acid':'benzene-1,3,5-tricarboxylic acid'
}
FAMILIES=['Bipyrazole/pyrazole','Carboxylate','Imidazolate/azolate','Pyridyl/N-donor','Phosphonate','Sulfonate','Curcumin/β-diketonate','Mixed donor','Other/unknown']
COUNTERIONS=['nitrate','acetate','chloride','bromide','iodide','sulfate','perchlorate','triflate','tetrafluoroborate','hexafluorophosphate','carbonate','hydroxide','oxide','alkoxide','acetylacetonate','other']

def normalize_ligand(text):
    t=' '.join(str(text or '').strip().split())
    return COMMON_LIGAND_ALIASES.get(t.lower(),t)

def infer_family(text):
    s=str(text or '').lower()
    if 'pyraz' in s or 'bpz' in s: return 'Bipyrazole/pyrazole'
    if any(k in s for k in ['carbox','benzoic','bdc','btc','fumar','terephthal','trimesic']): return 'Carboxylate'
    if 'imidazol' in s or 'triazol' in s or 'tetrazol' in s: return 'Imidazolate/azolate'
    if 'pyrid' in s or 'bipyrid' in s or 'phenanthrol' in s: return 'Pyridyl/N-donor'
    if 'phosphon' in s: return 'Phosphonate'
    if 'sulfon' in s: return 'Sulfonate'
    if 'curc' in s or 'diketon' in s: return 'Curcumin/β-diketonate'
    return 'Other/unknown'

def parse_salt(formula):
    s=str(formula or '').replace(' ', '').replace('•','·')
    hyd=0.0
    m=re.search(r'·([0-9]+(?:\.[0-9]+)?)H2O',s,re.I)
    if m: hyd=float(m.group(1))
    elif 'H2O' in s: hyd=1.0
    sl=s.lower()
    tests=[('nitrate','no3'),('acetate','oac'),('perchlorate','clo4'),('sulfate','so4'),('triflate','cf3so3'),('tetrafluoroborate','bf4'),('hexafluorophosphate','pf6'),('bromide','br'),('chloride','cl'),('hydroxide','oh'),('alkoxide','och3'),('acetylacetonate','acac')]
    an='other'
    for name,token in tests:
        if token in sl: an=name; break
    ox=np.nan
    roman={'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6}
    rm=re.search(r'\(([ivx]+)\)',sl)
    if rm: ox=roman.get(rm.group(1),np.nan)
    else:
        for pat,val in [(r'no3\)3',3),(r'cl3',3),(r'no3\)2',2),(r'cl2',2),(r'oac\)2',2),(r'cl4',4),(r'och3\)4',4)]:
            if re.search(pat,sl): ox=val; break
    return {'Hydration_Number':hyd,'Counterion_Class':an,'Oxidation_State':ox}

def build_row(values):
    row=dict(values)
    row['Legante']=normalize_ligand(row.get('Legante',''))
    row['Famiglia_Legante']=row.get('Famiglia_Legante') or infer_family(row['Legante'])
    row['Ligand_Text']=(row['Legante']+' '+row['Famiglia_Legante']).lower()
    parsed=parse_salt(row.get('Sale_Metallico',''))
    for k,v in parsed.items():
        if row.get(k) in (None,'', 'auto'): row[k]=v
    vals=METALS.get(str(row.get('Metallo','')), [np.nan,np.nan,np.nan,np.nan,'unknown',np.nan])
    row['Metal_Atomic_Number'],row['Metal_Atomic_Weight'],row['Metal_Group'],row['Metal_Period'],row['Metal_Block'],row['Metal_Electronegativity']=vals
    return pd.DataFrame([row])

def precursor_formula(metal, oxidation, counterion, hydration):
    charges={'nitrate':-1,'acetate':-1,'chloride':-1,'bromide':-1,'iodide':-1,'perchlorate':-1,'triflate':-1,'tetrafluoroborate':-1,'hexafluorophosphate':-1,'sulfate':-2,'carbonate':-2,'hydroxide':-1,'oxide':-2}
    symbols={'nitrate':'NO3','acetate':'OAc','chloride':'Cl','bromide':'Br','iodide':'I','perchlorate':'ClO4','triflate':'OTf','tetrafluoroborate':'BF4','hexafluorophosphate':'PF6','sulfate':'SO4','carbonate':'CO3','hydroxide':'OH','oxide':'O'}
    q=charges.get(counterion,-1); an=symbols.get(counterion,counterion)
    try: ox=int(oxidation)
    except: ox=2
    import math
    g=math.gcd(abs(ox),abs(q)); nm=abs(q)//g; na=abs(ox)//g
    mf=metal+(str(nm) if nm>1 else '')
    af=f'({an}){na}' if na>1 and len(an)>1 else an+(str(na) if na>1 else '')
    h=float(hydration or 0)
    return mf+af+(f'·{h:g}H2O' if h>0 else '')
