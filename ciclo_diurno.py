#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ciclo_diurno.py — ciclo diurno medio de uma variavel (media no dominio do Eta),
comparando jaci, XC50 e ERA5, separado por horizonte de previsao (D+n).

Fluxo:
  - calcula(): le os NetCDF do(s) modelo(s) + ERA5, acumula somas por
    (fonte, prazo, hora) e grava um CSV (ciclo_diurno_<var>.csv) para nao ter
    de reprocessar os dados raw ao replotar.
  - plota(): um painel por prazo; linhas jaci (azul, -), XC50 (vermelho, --),
    ERA5 (preto, :).

Otimizacoes:
  - media espacial por arquivo vetorizada (xarray .mean nas dims lat/lon);
  - serie de dominio do ERA5 lida UMA vez por processo (bbox + mean lazy);
  - paralelizacao opcional por rodadas com --jobs N (cada processo acumula um
    pedaco e as somas sao mescladas no fim). Cada processo abre a SUA propria
    referencia ERA5, entao use um --jobs modesto para nao saturar o Lustre.

Uso:
  python ciclo_diurno.py --var t2m --leads 1 3 5 7 --jobs 4
  python ciclo_diurno.py --var t2m --replot            # so replota do CSV
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


def _acumula_chunk(cfg, var, unidade, e5, leads_alvo, tarefas):
    """Acumula soma/cont de (fonte,lead,hora) para uma lista de tarefas
    (modelo, mspec, run). Abre a SUA propria referencia ERA5 (uma vez).
    Usado tanto no modo sequencial quanto em cada processo do modo paralelo."""
    ref = V.RefEra5(cfg["referencias"]["era5"]["arquivo"])
    bbox = None; serie_e5 = None; te5 = None
    soma = {}; cont = {}

    def add(fonte, lead, hora, val):
        if not np.isfinite(val):
            return
        k = (fonte, lead, int(hora))
        soma[k] = soma.get(k, 0.0) + val
        cont[k] = cont.get(k, 0) + 1

    for modelo, mspec, run in tarefas:
        arq = V._resolve_arq(cfg, mspec, modelo, run, "netcdf", mspec.get("padrao"))
        if arq is None:
            continue
        umod = mspec.get("unidade")
        init = datetime.strptime(run, "%Y%m%d%H")
        try:
            ds = N.abre(arq)
            v, latn, lonn, tn = N.nomes(ds, mspec.get("var"),
                                        prefer="tp2m|t2m|pslm|u10|v10")
            lats, lons = N.grade(ds, latn, lonn)
            tempos = pd.to_datetime(ds[tn].values) if tn and tn in ds else None
            if tempos is None:
                ds.close(); continue
            mserie = _media_dominio(ds[v], latn, lonn, tn)
            if umod:
                mserie = N.converte_unidade(mserie, umod, unidade)
        except Exception as ex:
            print(f"  [{run}] erro: {ex}"); continue
        ds.close()
        if bbox is None:
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
            j = int(np.argmin(np.abs(te5 - np.datetime64(valido))))
            if abs((serie_e5.index[j] - valido).total_seconds()) <= 3 * 3600:
                add("ERA5", lead, valido.hour, float(serie_e5.iloc[j]))
    ref.close()
    return soma, cont


def _worker_cd(payload):
    return _acumula_chunk(*payload)


def calcula(cfg, args, saida):
    comp = cfg["componentes"][args.var]
    unidade = comp.get("unidade", "")
    e5 = comp.get("era5", {})
    leads_alvo = set(args.leads)

    # lista de tarefas (modelo, mspec, run)
    tarefas = []
    for modelo, mspec in comp["modelos"].items():
        subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
        runs = V._lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"),
                                args.max_rodadas)
        print(f"[{args.var}] {modelo}: {len(runs)} rodadas")
        for run in runs:
            tarefas.append((modelo, mspec, run))

    soma = {}; cont = {}

    def _merge(sc):
        s, c = sc
        for k, v in s.items():
            soma[k] = soma.get(k, 0.0) + v
        for k, v in c.items():
            cont[k] = cont.get(k, 0) + v

    jobs = int(getattr(args, "jobs", 1) or 1)
    if jobs > 1 and len(tarefas) > 1:
        import multiprocessing as mp
        nj = min(jobs, len(tarefas))
        chunks = [tarefas[i::nj] for i in range(nj)]      # round-robin
        payloads = [(cfg, args.var, unidade, e5, leads_alvo, ch) for ch in chunks]
        print(f"  paralelo: {nj} processos ({len(tarefas)} tarefas)")
        with mp.Pool(nj) as pool:
            for sc in pool.map(_worker_cd, payloads):
                _merge(sc)
    else:
        print(f"  sequencial ({len(tarefas)} tarefas)")
        _merge(_acumula_chunk(cfg, args.var, unidade, e5, leads_alvo, tarefas))

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
    ap = argparse.ArgumentParser(
        description="Ciclo diurno medio (jaci/XC50/ERA5) por horizonte de previsao.")
    ap.add_argument("--config", default="config_verificacao.yaml")
    ap.add_argument("--var", default="t2m")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 3, 5, 7])
    ap.add_argument("--max-rodadas", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processos paralelos (0 = min(cpu,8)); cada um abre "
                         "sua propria ERA5, use valores modestos.")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args(argv)
    if args.jobs == 0:
        import multiprocessing as mp
        args.jobs = min(mp.cpu_count(), 8)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    saida = args.saida or cfg.get("saida", "resultados_uni")
    os.makedirs(saida, exist_ok=True)
    unidade = cfg["componentes"][args.var].get("unidade", "")
    csv = os.path.join(saida, f"ciclo_diurno_{args.var}.csv")
    if args.replot:
        if not os.path.isfile(csv):
            print(f"ERRO: CSV nao encontrado: {csv}."); return 1
        df = pd.read_csv(csv)
        if args.leads:
            df = df[df.lead.isin(args.leads)]
        plota(df, args.var, unidade, saida)
        return 0
    df, unidade = calcula(cfg, args, saida)
    plota(df, args.var, unidade, saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
