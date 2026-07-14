#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ciclo_diurno.py
===============

Ciclo diurno medio de uma variavel (padrao T2m): jaci, XC50 e ERA5 no MESMO
grafico, com um painel por prazo de previsao (D+1, D+2, ...).

Fluxo em dois passos (rapido de replotar):
  1. CALCULO  -> le os .nc sub-diarios (todas as horas) de cada modelo e o
     ERA5, tira a media espacial no dominio do Eta e agrega por
     (fonte, prazo, hora-do-dia). Grava ciclo_diurno_<var>.csv.
  2. FIGURA   -> plota do CSV. Use --replot para SO plotar (sem reprocessar).

Otimizacoes:
  - media espacial vetorizada por arquivo: da.mean(dim=(lat,lon)) de uma vez
    (em vez de fatia-a-fatia);
  - ERA5 lido UMA vez (serie da media na caixa), nao por tempo/por modelo.

Uso:
  python ciclo_diurno.py --config config_verificacao.yaml --var t2m --leads 1 3 5 7
  python ciclo_diurno.py --config config_verificacao.yaml --var t2m --replot
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import nucleo as N
import verifica as V

COR_FONTE = {"jaci": "#1f77b4", "xc50": "#d62728", "ERA5": "black"}
LS_FONTE = {"jaci": "-", "xc50": "--", "ERA5": ":"}
MK_FONTE = {"jaci": "o", "xc50": "s", "ERA5": "^"}


def _serie_era5_caixa(ref, var, bbox, de_unid, para_unid):
    """Serie temporal da media do ERA5 na caixa (bbox), lida UMA vez (lazy)."""
    da = ref.ds[var]
    la, lo = ref.lats, ref.lons
    ila = np.where((la >= bbox[2]) & (la <= bbox[3]))[0]
    ilo = np.where((lo >= bbox[0]) & (lo <= bbox[1]))[0]
    if ila.size == 0 or ilo.size == 0:
        ila = np.arange(la.size); ilo = np.arange(lo.size)
    dab = da.isel({ref.latn: ila, ref.lonn: ilo})
    # reduz so as dims espaciais; mantem o tempo
    dims_red = [d for d in dab.dims if d in (ref.latn, ref.lonn)]
    m = dab.mean(dim=dims_red, skipna=True).values.astype(float)
    if de_unid and para_unid:
        m = N.converte_unidade(m, de_unid, para_unid)
    return pd.Series(m, index=ref.tempos)


def _media_dominio(da, latn, lonn, tn):
    """Media espacial por tempo (1D), vetorizada. Reduz outras dims em 0."""
    sel = {d: 0 for d in da.dims if d not in (latn, lonn, tn)}
    d2 = da.isel(sel) if sel else da
    return d2.mean(dim=(latn, lonn), skipna=True).values.astype(float)


def calcula(cfg, args, saida):
    comp = cfg["componentes"][args.var]
    unidade = comp.get("unidade", "")
    e5 = comp.get("era5", {})
    leads_alvo = set(args.leads)

    ref = V.RefEra5(cfg["referencias"]["era5"]["arquivo"])
    bbox = None
    serie_e5 = None
    soma = {}; cont = {}

    def add(fonte, lead, hora, val):
        if not np.isfinite(val):
            return
        k = (fonte, lead, int(hora))
        soma[k] = soma.get(k, 0.0) + val
        cont[k] = cont.get(k, 0) + 1

    for modelo, mspec in comp["modelos"].items():
        subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
        runs = V._lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"),
                                args.max_rodadas)
        print(f"[{args.var}] {modelo}: {len(runs)} rodadas")
        umod = mspec.get("unidade")
        for ri, run in enumerate(runs, 1):
            arq = V._resolve_arq(cfg, mspec, modelo, run, "netcdf", mspec.get("padrao"))
            if arq is None:
                continue
            init = datetime.strptime(run, "%Y%m%d%H")
            try:
                ds = N.abre(arq)
                var, latn, lonn, tn = N.nomes(ds, mspec.get("var"),
                                              prefer="tp2m|t2m|pslm|u10|v10")
                lats, lons = N.grade(ds, latn, lonn)
                tempos = pd.to_datetime(ds[tn].values) if tn and tn in ds else None
                if tempos is None:
                    ds.close(); continue
                mserie = _media_dominio(ds[var], latn, lonn, tn)   # 1D, vetorizado
                if umod:
                    mserie = N.converte_unidade(mserie, umod, unidade)
            except Exception as ex:
                print(f"  [{run}] erro: {ex}"); continue
            ds.close()
            if bbox is None:                       # dominio do modelo + serie ERA5
                bbox = [float(lons.min()), float(lons.max()),
                        float(lats.min()), float(lats.max())]
                serie_e5 = _serie_era5_caixa(ref, e5.get("var"), bbox,
                                             e5.get("unidade"), unidade)
                te5 = serie_e5.index.values.astype("datetime64[ns]")
            for it, valido in enumerate(tempos):
                fh = (valido - pd.Timestamp(init)) / timedelta(hours=1)
                if fh <= 0:
                    continue
                lead = int(np.ceil(fh / 24.0))
                if lead not in leads_alvo or not V._periodo_ok(cfg, valido):
                    continue
                add(modelo, lead, valido.hour, mserie[it])
                # ERA5 no tempo valido: vizinho mais proximo na serie ja calculada
                j = int(np.argmin(np.abs(te5 - np.datetime64(valido))))
                if abs((serie_e5.index[j] - valido).total_seconds()) <= 3 * 3600:
                    add("ERA5", lead, valido.hour, float(serie_e5.iloc[j]))
            if ri % 20 == 0 or ri == len(runs):
                print(f"  {ri}/{len(runs)}")
    ref.close()

    df = pd.DataFrame([dict(fonte=k[0], lead=k[1], hora=k[2],
                            valor=soma[k] / cont[k], n=cont[k]) for k in soma])
    csv = os.path.join(saida, f"ciclo_diurno_{args.var}.csv")
    if not df.empty:
        df.sort_values(["fonte", "lead", "hora"]).to_csv(csv, index=False)
        print(f"CSV: {csv}")
    return df, unidade


def plota(df, var, unidade, saida):
    if df.empty:
        print("Sem dados para plotar."); return
    leads = sorted(df.lead.unique())
    ncol = min(3, len(leads)); nrow = int(np.ceil(len(leads) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.4 * nrow),
                            squeeze=False)
    fontes = [f for f in ["jaci", "xc50", "ERA5"] if f in df.fonte.unique()]
    for k, lead in enumerate(leads):
        ax = axs.ravel()[k]
        for fonte in fontes:
            sub = df[(df.lead == lead) & (df.fonte == fonte)].sort_values("hora")
            if sub.empty:
                continue
            ax.plot(sub.hora, sub.valor, LS_FONTE[fonte], marker=MK_FONTE[fonte],
                    color=COR_FONTE[fonte], lw=1.8, markersize=4, label=fonte)
        ax.set_title(f"D+{lead}"); ax.set_xlabel("Hora (UTC)")
        ax.set_ylabel(f"{var} ({unidade})"); ax.grid(alpha=0.3)
        ax.set_xticks(sorted(df.hora.unique()))
    for k in range(len(leads), nrow * ncol):
        axs.ravel()[k].axis("off")
    h = [plt.Line2D([0], [0], color=COR_FONTE[f], linestyle=LS_FONTE[f],
                    marker=MK_FONTE[f], label=f) for f in fontes]
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.99),
               ncol=len(h), fontsize=10, framealpha=0.9)
    fig.suptitle(f"Ciclo diurno medio - {var} (media no dominio do Eta) por prazo",
                 fontsize=13, x=0.5, y=0.995)
    cam = os.path.join(saida, f"ciclo_diurno_{var}.png")
    fig.savefig(cam, dpi=130); plt.close(fig)
    print(f"Figura: {cam}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ciclo diurno medio (jaci/XC50/ERA5) por prazo.")
    ap.add_argument("--config", default="config_verificacao.yaml")
    ap.add_argument("--var", default="t2m", help="componente da config (t2m, pnmm, ...)")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7],
                    help="prazos (dias) a plotar, um painel por prazo")
    ap.add_argument("--max-rodadas", type=int, default=0)
    ap.add_argument("--saida", default=None, help="pasta de saida (padrao: cfg.saida)")
    ap.add_argument("--replot", action="store_true",
                    help="SO plota, lendo o CSV ja gerado (sem reprocessar)")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    saida = args.saida or cfg.get("saida", "resultados_uni")
    os.makedirs(saida, exist_ok=True)
    unidade = cfg["componentes"][args.var].get("unidade", "")
    csv = os.path.join(saida, f"ciclo_diurno_{args.var}.csv")

    if args.replot:
        if not os.path.isfile(csv):
            print(f"ERRO: CSV nao encontrado: {csv}. Rode sem --replot primeiro.")
            return 1
        df = pd.read_csv(csv)
        # respeita --leads no replot, se informado
        if args.leads:
            df = df[df.lead.isin(args.leads)]
        plota(df, args.var, unidade, saida)
        return 0

    df, unidade = calcula(cfg, args, saida)
    plota(df, args.var, unidade, saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
