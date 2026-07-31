from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from .chem import build_row
ROOT=Path(__file__).resolve().parents[1]
ART=joblib.load(ROOT/'models/MOF_ChemAware_Ensemble_v8_0.joblib')
SCHEMA=json.loads((ROOT/'models/feature_schema_v8_0.json').read_text())
DB=pd.read_csv(ROOT/'data/knowledge_database.csv')
FEATURES=ART['features']

def predict(values):
    x=build_row(values)
    for c in FEATURES:
        if c not in x: x[c]=np.nan
    x=x[FEATURES]
    p=ART['weights'][0]*ART['rf_model'].predict_proba(x)+ART['weights'][1]*ART['ligand_text_model'].predict_proba(x)
    return x, p[0], int(np.argmax(p[0]))

def applicability(values):
    ligand=str(values.get('Legante','')).strip().lower(); metal=str(values.get('Metallo','')); salt=str(values.get('Sale_Metallico',''))
    seen_lig=ligand in set(DB['Legante'].astype(str).str.lower())
    seen_metal=metal in set(DB['Metallo'].astype(str))
    seen_salt=salt in set(DB['Sale_Metallico'].astype(str))
    score=0.50*seen_lig+0.30*seen_metal+0.20*seen_salt
    if score>=0.8: label='Inside domain'
    elif score>=0.3: label='Intermediate / partial extrapolation'
    else: label='Outside domain'
    return {'score':float(score),'label':label,'ligand_seen':seen_lig,'metal_seen':seen_metal,'salt_seen':seen_salt}

def similar(values,n=15):
    d=DB.copy(); metal=str(values.get('Metallo','')); fam=str(values.get('Famiglia_Legante',''))
    d['_score']=0
    d.loc[d['Metallo'].astype(str)==metal,'_score']+=3
    d.loc[d['Famiglia_Legante'].astype(str)==fam,'_score']+=2
    for c,w in [('Temperatura_C',1),('Tempo_ore',1),('Rapporto_LM',1)]:
        try:
            scale=max(float(d[c].std()),1); val=float(values.get(c,np.nan)); d['_score']+=np.exp(-abs(pd.to_numeric(d[c],errors='coerce')-val)/scale)*w
        except: pass
    return d.sort_values('_score',ascending=False).head(n)

def explain_prediction(values):
    """Local, model-based sensitivity summary for the current synthesis.

    Each condition is varied independently across plausible values observed in the
    experimental database. The output is descriptive rather than causal: it shows
    which editable parameters can most change P(crystalline) near the current input.
    """
    _, base_p, _ = predict(values)
    base_cryst = float(base_p[2])
    specs = {
        'Solvente': list(DB['Solvente'].dropna().astype(str).value_counts().head(10).index),
        'Temperatura_C': sorted(set([max(20.0, float(values.get('Temperatura_C',120))+d) for d in (-40,-20,20,40)] + [80.0,100.0,120.0,150.0])),
        'Tempo_ore': sorted(set([max(0.5, float(values.get('Tempo_ore',24))*m) for m in (0.5,2.0)] + [6.0,12.0,24.0,48.0,72.0])),
        'Rapporto_LM': sorted(set([max(0.1, float(values.get('Rapporto_LM',1))+d) for d in (-1.0,-0.5,0.5,1.0)] + [0.5,1.0,2.0,3.0])),
        'Additivo_Colinker': list(DB['Additivo_Colinker'].fillna('Nessuno').astype(str).value_counts().head(8).index),
        'Volume solvente': sorted(set([max(0.5, float(values.get('Volume solvente',10))*m) for m in (0.5,1.5,2.0)])),
    }
    labels = {
        'Solvente':'Solvent', 'Temperatura_C':'Temperature', 'Tempo_ore':'Reaction time',
        'Rapporto_LM':'Ligand/metal ratio', 'Additivo_Colinker':'Additive / co-linker',
        'Volume solvente':'Solvent volume'
    }
    rows=[]
    for field, candidates in specs.items():
        tested=[]
        current=values.get(field)
        for candidate in candidates:
            if str(candidate)==str(current):
                continue
            v=dict(values); v[field]=candidate
            try:
                _, pp, _=predict(v); tested.append((candidate,float(pp[2])))
            except Exception:
                continue
        if not tested:
            continue
        best_val,best_p=max(tested,key=lambda z:z[1])
        mean_alt=float(np.mean([z[1] for z in tested]))
        improvement=best_p-base_cryst
        support=base_cryst-mean_alt
        direction='Limiting' if improvement>0.025 else ('Favorable' if support>0.025 else 'Neutral')
        influence=max(abs(improvement),abs(support))
        rows.append({
            'Parameter':labels[field], 'Field':field, 'Current':current,
            'Influence':influence, 'Direction':direction,
            'Potential_improvement':max(0.0,improvement),
            'Best_alternative':best_val, 'Best_P_crystalline':best_p,
        })
    return pd.DataFrame(rows).sort_values('Influence',ascending=False).reset_index(drop=True), base_cryst


def optimize(values,top_n=10):
    base=dict(values); temps=sorted(set([max(20,float(base['Temperatura_C'])+x) for x in (-40,-20,0,20,40)]))
    times=sorted(set([max(0.5,float(base['Tempo_ore'])*x) for x in (0.5,1,2)]))
    ratios=sorted(set([max(0.1,float(base['Rapporto_LM'])+x) for x in (-1,-0.5,0,0.5,1)]))
    solvents=list(DB['Solvente'].dropna().astype(str).value_counts().head(8).index)
    raw=[]
    for t in temps:
      for h in times:
       for r in ratios:
        for solv in solvents:
         v=dict(base); v.update({'Temperatura_C':t,'Tempo_ore':h,'Rapporto_LM':r,'Solvente':solv}); raw.append(v)
    engineered=pd.concat([build_row(v) for v in raw],ignore_index=True)
    for c in FEATURES:
        if c not in engineered: engineered[c]=np.nan
    x=engineered[FEATURES]
    probs=ART['weights'][0]*ART['rf_model'].predict_proba(x)+ART['weights'][1]*ART['ligand_text_model'].predict_proba(x)
    ad=applicability(base); ad_score=ad['score']
    out=pd.DataFrame(raw)
    out['P_Failed']=probs[:,0]; out['P_Amorphous']=probs[:,1]; out['P_Crystalline']=probs[:,2]
    out['AD_score']=ad_score; out['Optimized_score']=out['P_Crystalline']*(0.65+0.35*ad_score)
    return out.sort_values(['Optimized_score','P_Crystalline'],ascending=False).drop_duplicates(['Temperatura_C','Tempo_ore','Rapporto_LM','Solvente']).head(top_n)
