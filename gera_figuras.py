#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gera_figuras.py — regenera TODAS as figuras a partir do binario verificacao.pkl
gravado por verifica.py, sem reprocessar os dados raw.

Para cada componente:
  - fig_continuas_dominio_<comp>.png ... BIAS/MAE/RMSE/SCORR x prazo
        (Todo/Continente/Oceano).
  - fig_continuas_regioes_<comp>.png ... idem para Amazonia/Nordeste/Sudeste/Sul.
  - fig_ETS_<comp>.png / fig_BIAScat_<comp>.png ... so precipitacao (x limiar).
  - fig_FSS_<comp>_<escala>.png ........ so precipitacao (FSS x limiar por prazo).
  - mapas_<comp>_<modelo>.png .......... obs/prev/vies medios.

Uso:
  python gera_figuras.py --binario resultados_uni/verificacao.pkl
  python gera_figuras.py --binario resultados_uni/verificacao.pkl --componentes t2m
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

import nucleo as N


def figuras_componente(comp, saida):
    nome = comp["nome"]; unidade = comp.get("unidade", "")
    regd = comp.get("regioes_dominio", ["Todo"])
    dfc = comp["continuas"]
    if dfc is not None and not dfc.empty:
        N.plota_continuas_regiao(
            dfc, [r for r in regd if r in dfc.regiao.unique()], unidade,
            f"{nome}: BIAS/MAE/RMSE/SCORR x prazo (dominio)",
            os.path.join(saida, f"fig_continuas_dominio_{nome}.png"))
        dom = set(regd) | {"Todo", "Continente", "Oceano"}
        regbr = [r for r in dfc.regiao.unique() if r not in dom]   # bacias hidrog.
        if regbr:
            N.plota_continuas_regiao(
                dfc, regbr, unidade,
                f"{nome}: BIAS/MAE/RMSE/SCORR x prazo (grandes bacias hidrograficas)",
                os.path.join(saida, f"fig_continuas_regioes_{nome}.png"))

    dfk = comp.get("categoricas")
    if dfk is not None and not dfk.empty:
        N.plota_por_prazo(dfk, "ETS", "ETS", f"{nome}: ETS x limiar por prazo",
                          os.path.join(saida, f"fig_ETS_{nome}.png"))
        N.plota_por_prazo(dfk, "FBIAS", "BIAS",
                          f"{nome}: BIAS categorico x limiar por prazo",
                          os.path.join(saida, f"fig_BIAScat_{nome}.png"), ref1=True)

    dff = comp.get("fss")
    if dff is not None and not dff.empty:
        for esc in sorted(dff.escala_px.unique()):
            sub = dff[dff.escala_px == esc].rename(columns={"FSS": "FSS"})
            N.plota_por_prazo(sub, "FSS", "FSS",
                              f"{nome}: FSS (vizinhanca {esc} px) x limiar por prazo",
                              os.path.join(saida, f"fig_FSS_{nome}_{esc}px.png"))

    # ---- mapas espaciais: media diaria por (modelo, prazo), colunas Todo/meses
    grade = comp.get("grade", {})
    difcmap = "BrBG" if comp.get("tipo") == "acum24" else "RdBu_r"
    ref_nome = "MERGE" if comp.get("tipo") == "acum24" else "ERA5"
    mapas = comp.get("mapas", {})
    chaves = list(mapas.keys())
    if chaves and isinstance(chaves[0], tuple):
        modelos = sorted({k[0] for k in chaves})
        leads = sorted({k[1] for k in chaves})
        # escala de cor UNICA por variavel (comum a jaci/XC50/referencia)
        vmin, vmax, bmax = N.escala_global(mapas)
        for modelo in modelos:
            for lead in leads:
                pormes = {k[2]: mapas[k] for k in chaves
                          if k[0] == modelo and k[1] == lead
                          and np.asarray(mapas[k]["n"]).sum() > 0}
                if not pormes:
                    continue
                N.plota_mapas_mes(
                    grade["lats"], grade["lons"], pormes, unidade,
                    f"{nome} - {modelo} - D+{lead} (media diaria por periodo)",
                    os.path.join(saida, f"mapa_{nome}_{modelo}_D{lead}.png"),
                    difcmap=difcmap, ref_nome=ref_nome,
                    vminmax=(vmin, vmax), bmax_fixo=bmax)
    else:                                      # compat: formato antigo
        for modelo, arrs in mapas.items():
            if "sdif" in arrs and np.asarray(arrs["n"]).sum() > 0:
                N.plota_mapas(arrs, grade["lats"], grade["lons"], unidade,
                              f"{nome} - {modelo}",
                              os.path.join(saida, f"mapas_{nome}_{modelo}.png"),
                              difcmap=difcmap)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Regenera figuras a partir do binario verificacao.pkl.")
    ap.add_argument("--binario", default="resultados_uni/verificacao.pkl")
    ap.add_argument("--figuras", default=None)
    ap.add_argument("--componentes", nargs="*", default=None)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.binario):
        print(f"ERRO: binario nao encontrado: {args.binario}"); return 1
    with open(args.binario, "rb") as f:
        res = pickle.load(f)
    saida = args.figuras or os.path.dirname(os.path.abspath(args.binario))
    os.makedirs(saida, exist_ok=True)

    escolhidos = args.componentes or list(res.keys())
    for nome in escolhidos:
        if nome not in res:
            print(f"[aviso] componente '{nome}' nao esta no binario"); continue
        comp = dict(res[nome]); comp["nome"] = nome
        # tabelas podem vir como dict-of-lists (binario portavel) -> DataFrame
        for key in ("continuas", "categoricas", "fss"):
            v = comp.get(key)
            if isinstance(v, dict):
                comp[key] = pd.DataFrame(v)
        print(f"Componente {nome}:")
        figuras_componente(comp, saida)
    print(f"\nFiguras em: {os.path.abspath(saida)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
