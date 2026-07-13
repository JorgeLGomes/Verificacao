#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifica_eta.py
===============

Verificacao de precipitacao acumulada em 24 h do modelo Eta (CPTEC) contra a
base observada MERGE, na resolucao do observado (10 km).

Estrutura de dados esperada (padrao do FTP grpeta/Verif-prec):

    <BASE>/
      nc/                 rodadas "jaci"  (precip em METROS)  -> x1000 = mm
        YYYYMMDDHH/                       (uma pasta por rodada)
          PREC-ACUM24h_YYYYMMDD.nc        (ja acumulado em 24 h; USADO)
          PREC_YYYYMMDD.nc                (bruto; ignorado por padrao)
      nc_oper/            rodadas XC50    (precip em MILIMETROS)
        YYYYMMDDHH/
          PREC_YYYYMMDD.nc                (sub-diario; ACUMULADO em 24 h aqui)
      nc_merge/           base observada  (MILIMETROS, 10 km)
        MERGE_*.nc                        (serie temporal cobrindo o periodo)

O que o script faz
------------------
  1. Percorre cada rodada de 'nc' (jaci) e 'nc_oper' (XC50).
  2. Constroi os campos de precipitacao acumulada em 24 h da previsao:
       - jaci ....... usa o PREC-ACUM24h e converte m -> mm (x1000);
       - XC50 ....... acumula os passos sub-diarios de PREC em janelas de 24 h.
     Duas janelas por padrao: 12Z-12Z e 00Z-00Z (configuravel).
  3. Para cada janela/prazo, obtem do MERGE o acumulado de 24 h da MESMA janela.
  4. Regrida a previsao (8 km) para a grade do MERGE (10 km) - verificacao a 10 km.
  5. Calcula metricas continuas, categoricas (por limiar) e FSS.
  6. Agrega e salva:
       - placar_por_modelo.csv ...... scores agregados por modelo e janela;
       - scores_por_prazo.csv ....... scores em funcao do prazo (D+1, D+2, ...);
       - scores_por_rodada.csv ...... uma linha por rodada/prazo (serie temporal);
       - mapas_<modelo>.png ......... vies medio, erro medio e campos medios.

Uso
---
    # inspecionar a estrutura interna de um arquivo (nomes de var/coords/tempo):
    python verifica_eta.py --inspecionar /caminho/PREC-ACUM24h_20260101.nc

    # verificacao completa:
    python verifica_eta.py \
        --base /caminho/Verif-prec \
        --saida resultados_eta \
        --limiares 1 10 25 50 --escalas-fss 1 3 5 \
        --janelas 12 0 --max-rodadas 0

Depois de rodar --inspecionar, confira se os nomes detectados batem; se preciso,
ajuste o dicionario CONFIG (nomes de variavel e fator de unidade) no topo.

Dependencias: numpy pandas xarray scipy netCDF4 matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIGURACAO DOS MODELOS
# ---------------------------------------------------------------------------
# fator .... multiplicador para converter a unidade do arquivo para mm.
# var ...... nome da variavel de precip (None = detecta automaticamente).
# padrao ... glob do arquivo de precip dentro da pasta da rodada.
# acumulado. True  -> arquivo ja traz o acumulado de 24 h (so seleciona/relabela);
#            False -> precip sub-diaria, o script soma os passos em 24 h.
CONFIG = {
    "jaci": {
        "subdir": "nc",
        "padrao": "PREC-ACUM24h_*.nc",
        "var": None,
        "fator": 1000.0,      # metros -> mm
        "acumulado": True,
    },
    "xc50": {
        "subdir": "nc_oper",
        "padrao": "PREC_*.nc",
        "var": None,
        "fator": 1.0,         # ja em mm
        "acumulado": False,   # acumular 24 h a partir dos passos
    },
}
MERGE_SUBDIR = "nc_merge"
MERGE_PADRAO = "MERGE_*.nc"
MERGE_FATOR = 1.0             # ja em mm
MERGE_VAR = None              # detecta automaticamente

# nomes candidatos de coordenadas
CAND_LAT = ["lat", "latitude", "y", "Y", "XLAT", "g0_lat_0", "rlat"]
CAND_LON = ["lon", "longitude", "x", "X", "XLONG", "g0_lon_1", "rlon"]
CAND_TIME = ["time", "t", "valid_time", "Time", "times", "tempo"]


# ===========================================================================
# NUCLEO DE METRICAS (comum a qualquer verificacao)
# ===========================================================================
@dataclass
class Campo:
    dados: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    nome: str = ""

    def __post_init__(self):
        self.dados = np.asarray(self.dados, dtype=float)
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)


def _norm_lons(lons):
    lons = np.asarray(lons, dtype=float)
    return np.where(lons > 180.0, lons - 360.0, lons)


def regrid(origem: Campo, destino: Campo, metodo: str = "linear") -> Campo:
    """Interpola 'origem' para a grade de 'destino' (bilinear ou nearest)."""
    from scipy.interpolate import RegularGridInterpolator
    lats_o, dados = origem.lats, origem.dados
    if lats_o[0] > lats_o[-1]:
        lats_o = lats_o[::-1]
        dados = dados[::-1, :]
    lons_o = origem.lons
    if lons_o[0] > lons_o[-1]:
        lons_o = lons_o[::-1]
        dados = dados[:, ::-1]
    interp = RegularGridInterpolator((lats_o, lons_o), dados, method=metodo,
                                     bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(destino.lons, destino.lats)
    z = interp(np.column_stack([LAT.ravel(), LON.ravel()])).reshape(LAT.shape)
    return Campo(z, destino.lats, destino.lons, nome=origem.nome + "_regrid")


# --- acumuladores de estatisticas suficientes (permitem agregacao correta) ---
class AccContinuo:
    def __init__(self):
        self.n = 0.0
        self.sp = self.so = self.spp = self.soo = self.spo = 0.0
        self.sdif = self.sabs = self.sdif2 = 0.0

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        p, o = p[m], o[m]
        d = p - o
        self.n += p.size
        self.sp += p.sum(); self.so += o.sum()
        self.spp += (p * p).sum(); self.soo += (o * o).sum()
        self.spo += (p * o).sum()
        self.sdif += d.sum(); self.sabs += np.abs(d).sum()
        self.sdif2 += (d * d).sum()

    def scores(self):
        n = self.n
        if n == 0:
            return dict(n=0, bias=np.nan, mae=np.nan, rmse=np.nan, corr=np.nan,
                        media_prev=np.nan, media_obs=np.nan, vies_mult=np.nan)
        mp, mo = self.sp / n, self.so / n
        cov = self.spo / n - mp * mo
        vp = self.spp / n - mp * mp
        vo = self.soo / n - mo * mo
        corr = cov / np.sqrt(vp * vo) if vp > 0 and vo > 0 else np.nan
        return dict(n=int(n), bias=self.sdif / n, mae=self.sabs / n,
                    rmse=np.sqrt(self.sdif2 / n), corr=corr,
                    media_prev=mp, media_obs=mo,
                    vies_mult=(self.sp / self.so) if self.so > 0 else np.nan)


class AccCategoria:
    """Acumula tabela de contingencia por limiar."""
    def __init__(self, limiares):
        self.limiares = list(limiares)
        self.abcd = {t: [0, 0, 0, 0] for t in limiares}  # a=hit b=fa c=miss d=cn

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        p, o = p[m], o[m]
        for t in self.limiares:
            pp, oo = p >= t, o >= t
            self.abcd[t][0] += int(np.sum(pp & oo))
            self.abcd[t][1] += int(np.sum(pp & ~oo))
            self.abcd[t][2] += int(np.sum(~pp & oo))
            self.abcd[t][3] += int(np.sum(~pp & ~oo))

    def scores(self):
        linhas = []
        for t in self.limiares:
            a, b, c, d = self.abcd[t]
            n = a + b + c + d
            sd = lambda x, y: x / y if y else np.nan
            a_ref = sd((a + b) * (a + c), n)
            esp = sd((a + b) * (a + c) + (c + d) * (b + d), n)
            linhas.append(dict(
                limiar_mm=t, hits=a, false_alarms=b, misses=c, correct_neg=d,
                n=n, POD=sd(a, a + c), FAR=sd(b, a + b), POFD=sd(b, b + d),
                CSI=sd(a, a + b + c), FBIAS=sd(a + b, a + c),
                ETS=sd(a - a_ref, a + b + c - a_ref),
                HSS=sd(a + d - esp, n - esp), acuracia=sd(a + d, n)))
        return linhas


class AccFSS:
    """Acumula numerador/denominador do FSS por (limiar, escala)."""
    def __init__(self, limiares, escalas):
        self.limiares = list(limiares)
        self.escalas = list(escalas)
        self.num = {}
        self.den = {}
        self.cnt = {}
        for t in limiares:
            for s in escalas:
                self.num[(t, s)] = 0.0
                self.den[(t, s)] = 0.0
                self.cnt[(t, s)] = 0.0

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        for t in self.limiares:
            pb = (np.where(m, p, 0.0) >= t) & m
            ob = (np.where(m, o, 0.0) >= t) & m
            for s in self.escalas:
                n = s // 2
                fp = _fracoes(pb.astype(float), n)[m]
                fo = _fracoes(ob.astype(float), n)[m]
                self.num[(t, s)] += np.sum((fp - fo) ** 2)
                self.den[(t, s)] += np.sum(fp ** 2) + np.sum(fo ** 2)
                self.cnt[(t, s)] += fp.size

    def scores(self):
        linhas = []
        for t in self.limiares:
            for s in self.escalas:
                den = self.den[(t, s)]
                fss = 1.0 - self.num[(t, s)] / den if den > 0 else np.nan
                linhas.append(dict(limiar_mm=t, escala_px=s, FSS=fss,
                                   n=int(self.cnt[(t, s)])))
        return linhas


def _fracoes(campo_bin, n):
    """Fracao de pixels excedentes na vizinhanca quadrada (2n+1). Vetorizado."""
    if n <= 0:
        return np.asarray(campo_bin, dtype=float)
    from scipy.ndimage import uniform_filter
    return uniform_filter(np.asarray(campo_bin, dtype=float),
                          size=2 * n + 1, mode="constant", cval=0.0)


def scores_par_continuo(p, o):
    """Scores continuos de UM par (para a serie por rodada/prazo)."""
    acc = AccContinuo(); acc.add(p, o)
    return acc.scores()


# ===========================================================================
# LEITURA / TEMPO
# ===========================================================================
def _acha(ds, cands):
    for c in cands:
        if c in ds.coords or c in ds.variables or c in getattr(ds, "dims", {}):
            return c
    return None


def _abre(caminho):
    import xarray as xr
    return xr.open_dataset(caminho, decode_times=True)


def _nomes(ds, var):
    latn = _acha(ds, CAND_LAT)
    lonn = _acha(ds, CAND_LON)
    tn = _acha(ds, CAND_TIME)
    if var is None:
        cand = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if not cand:
            raise ValueError("nenhuma variavel 2D+ encontrada")
        # prefere nome contendo prec/pr
        pref = [v for v in cand if re.search(r"prec|pr|rain|acpc|tp", v, re.I)]
        var = pref[0] if pref else cand[0]
    return var, latn, lonn, tn


def inspeciona(caminho):
    ds = _abre(caminho)
    print(f"\n=== {caminho} ===")
    print("Dimensoes:", dict(ds.sizes))
    print("Coordenadas:", list(ds.coords))
    print("Variaveis:", list(ds.data_vars))
    var, latn, lonn, tn = _nomes(ds, None)
    print(f"Detectado -> var='{var}'  lat='{latn}'  lon='{lonn}'  time='{tn}'")
    if tn and tn in ds:
        tv = pd.to_datetime(ds[tn].values)
        print(f"Tempo: {len(tv)} passos | {tv[0]} .. {tv[-1]}")
        if len(tv) > 1:
            dt = pd.Series(tv).diff().dropna().mode()
            print(f"Passo temporal (moda): {dt.iloc[0] if len(dt) else '?'}")
    if latn and lonn:
        la, lo = ds[latn].values, ds[lonn].values
        print(f"Lat: {la.min():.3f}..{la.max():.3f} ({la.size})  "
              f"Lon: {_norm_lons(lo).min():.3f}..{_norm_lons(lo).max():.3f} ({lo.size})")
    if var in ds:
        u = ds[var].attrs.get("units", "?")
        v = ds[var].values
        print(f"'{var}': unidades='{u}'  min={np.nanmin(v):.4g}  max={np.nanmax(v):.4g}")
    ds.close()


def _da_2d_no_tempo(da, latn, lonn, tn, it):
    """Extrai o campo 2D (lat,lon) de um DataArray no indice de tempo 'it'."""
    sel = {}
    for d in da.dims:
        if d in (latn, lonn):
            continue
        sel[d] = it if d == tn else 0
    da2 = da.isel(sel) if sel else da
    return da2.transpose(latn, lonn).values


def _grade(ds, latn, lonn):
    return ds[latn].values, _norm_lons(ds[lonn].values)


# ===========================================================================
# CONSTRUCAO DOS ACUMULADOS DE 24 h
# ===========================================================================
@dataclass
class Janela24:
    fim: pd.Timestamp        # instante final da janela (fim do acumulado)
    hora: int                # 12 ou 0
    campo: Campo             # acumulado 24 h (mm) na grade nativa
    lead: int                # prazo em dias (D+lead)
    prazo_h: int             # prazo em horas (fim da janela - init) - sem ambiguidade


def _lead_dias(tfim, init):
    """Prazo em dias (D+n) e em horas para a janela terminando em tfim.
    Usa piso: janela 00Z->24h e janela 12Z->36h ambas caem em D+1, etc."""
    h = (pd.Timestamp(tfim) - pd.Timestamp(init)) / timedelta(hours=1)
    return int(h // 24), int(round(h))


def previsao_acum24(caminho, cfg, horas_janela, init):
    """Retorna lista de Janela24 para um arquivo de previsao."""
    ds = _abre(caminho)
    var, latn, lonn, tn = _nomes(ds, cfg["var"])
    lats, lons = _grade(ds, latn, lonn)
    da = ds[var]
    fator = cfg["fator"]
    saida = []

    if tn is None or tn not in ds or ds[tn].size == 1:
        # campo unico ja acumulado
        campo = Campo(_da_2d_no_tempo(da, latn, lonn, tn, 0) * fator, lats, lons,
                      nome=cfg["subdir"])
        tfim = pd.Timestamp(init) + timedelta(hours=24)
        saida.append(Janela24(tfim, tfim.hour, campo, lead=1, prazo_h=24))
        ds.close()
        return saida

    tempos = pd.to_datetime(ds[tn].values)

    if cfg["acumulado"]:
        # cada passo ja e um acumulado de 24 h valido no seu instante
        for it, t in enumerate(tempos):
            if t.hour not in horas_janela:
                continue
            campo = Campo(_da_2d_no_tempo(da, latn, lonn, tn, it) * fator,
                          lats, lons, nome=cfg["subdir"])
            lead, ph = _lead_dias(t, init)
            if lead < 1:
                continue
            saida.append(Janela24(t, t.hour, campo, lead, ph))
        ds.close()
        return saida

    # precip sub-diaria: acumula 24 h em cada janela pedida
    dt_h = _passo_horas(tempos)
    passos_por_dia = int(round(24 / dt_h)) if dt_h else None
    for hora in horas_janela:
        fins = [t for t in tempos if t.hour == hora]
        for tfim in fins:
            tini = tfim - timedelta(hours=24)
            mask = (tempos > tini) & (tempos <= tfim)
            idx = np.where(mask)[0]
            if passos_por_dia and len(idx) < passos_por_dia:
                continue  # janela incompleta
            soma = None
            for it in idx:
                c = _da_2d_no_tempo(da, latn, lonn, tn, int(it)) * fator
                soma = c if soma is None else soma + c
            if soma is None:
                continue
            lead, ph = _lead_dias(tfim, init)
            if lead < 1:
                continue
            saida.append(Janela24(tfim, hora, Campo(soma, lats, lons,
                                                    nome=cfg["subdir"]), lead, ph))
    ds.close()
    return saida


def _passo_horas(tempos):
    if len(tempos) < 2:
        return None
    difs = pd.Series(tempos).diff().dropna()
    return difs.dt.total_seconds().mode().iloc[0] / 3600.0


# ===========================================================================
# OBSERVACAO MERGE
# ===========================================================================
class MergeObs:
    """Le o MERGE uma vez e fornece o acumulado de 24 h por janela."""
    def __init__(self, caminho):
        self.ds = _abre(caminho)
        self.var, self.latn, self.lonn, self.tn = _nomes(self.ds, MERGE_VAR)
        self.lats, self.lons = _grade(self.ds, self.latn, self.lonn)
        self.tempos = (pd.to_datetime(self.ds[self.tn].values)
                       if self.tn and self.tn in self.ds else None)
        self.dt_h = _passo_horas(self.tempos) if self.tempos is not None else None
        self.grade = Campo(np.zeros((self.lats.size, self.lons.size)),
                           self.lats, self.lons, nome="merge_grade")

    def acum24(self, tfim: pd.Timestamp) -> Optional[Campo]:
        """Acumulado de 24 h do MERGE terminando em tfim (mm)."""
        da = self.ds[self.var]
        if self.tempos is None:
            arr = _da_2d_no_tempo(da, self.latn, self.lonn, self.tn, 0)
            return Campo(arr * MERGE_FATOR, self.lats, self.lons, nome="merge")
        # dados diarios: um passo por dia (~24 h) -> uso direto
        if self.dt_h and self.dt_h >= 23:
            dif = np.abs((self.tempos - tfim).total_seconds())
            i = int(np.argmin(dif))
            if dif[i] > 6 * 3600:      # sem passo proximo o suficiente
                return None
            arr = _da_2d_no_tempo(da, self.latn, self.lonn, self.tn, i)
            return Campo(arr * MERGE_FATOR, self.lats, self.lons, nome="merge")
        # dados sub-diarios: soma na janela (tfim-24h, tfim]
        tini = tfim - timedelta(hours=24)
        mask = (self.tempos > tini) & (self.tempos <= tfim)
        idx = np.where(mask)[0]
        if self.dt_h:
            if len(idx) < int(round(24 / self.dt_h)):
                return None
        elif len(idx) == 0:
            return None
        soma = None
        for it in idx:
            c = _da_2d_no_tempo(da, self.latn, self.lonn, self.tn, int(it))
            soma = c if soma is None else soma + c
        return Campo(soma * MERGE_FATOR, self.lats, self.lons, nome="merge")

    def close(self):
        self.ds.close()


# ===========================================================================
# PIPELINE
# ===========================================================================
def lista_rodadas(base, subdir, limite=0):
    d = os.path.join(base, subdir)
    if not os.path.isdir(d):
        return []
    runs = sorted(x for x in os.listdir(d)
                  if re.fullmatch(r"\d{10}", x) and
                  os.path.isdir(os.path.join(d, x)))
    return runs[:limite] if limite else runs


def acha_arquivo_prev(base, subdir, run, padrao):
    p = os.path.join(base, subdir, run, padrao)
    achados = sorted(glob.glob(p))
    return achados[0] if achados else None


class Mapas:
    """Acumula somas por celula (grade do MERGE) para mapas medios por modelo."""
    def __init__(self, grade: Campo):
        s = (grade.lats.size, grade.lons.size)
        self.grade = grade
        self.sp = np.zeros(s); self.so = np.zeros(s)
        self.sdif = np.zeros(s); self.n = np.zeros(s)

    def add(self, p, o):
        m = np.isfinite(p) & np.isfinite(o)
        self.sp[m] += p[m]; self.so[m] += o[m]
        self.sdif[m] += (p[m] - o[m]); self.n[m] += 1

    def salva(self, caminho, titulo):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  [aviso] matplotlib indisponivel ({e})")
            return
        with np.errstate(invalid="ignore"):
            mp = np.where(self.n > 0, self.sp / self.n, np.nan)
            mo = np.where(self.n > 0, self.so / self.n, np.nan)
            mb = np.where(self.n > 0, self.sdif / self.n, np.nan)
        if self.grade.lats[0] > self.grade.lats[-1]:
            mp, mo, mb = mp[::-1], mo[::-1], mb[::-1]
        ext = [self.grade.lons.min(), self.grade.lons.max(),
               self.grade.lats.min(), self.grade.lats.max()]
        vmax = np.nanpercentile(np.concatenate(
            [mo[np.isfinite(mo)], mp[np.isfinite(mp)]]), 99) if \
            np.isfinite(mo).any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        bmax = np.nanpercentile(np.abs(mb[np.isfinite(mb)]), 99) if \
            np.isfinite(mb).any() else 1.0
        if not np.isfinite(bmax) or bmax <= 0:
            bmax = 1.0
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        for a, campo, tit, cmap, vlim in [
            (ax[0], mo, "Obs media (MERGE)", "Blues", (0, vmax)),
            (ax[1], mp, "Prev media", "Blues", (0, vmax)),
            (ax[2], mb, "Vies medio (Prev-Obs)", "RdBu_r", (-bmax, bmax)),
        ]:
            im = a.imshow(campo, origin="lower", extent=ext, aspect="auto",
                          cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            a.set_title(tit); a.set_xlabel("lon"); a.set_ylabel("lat")
            fig.colorbar(im, ax=a, shrink=0.85, label="mm")
        fig.suptitle(titulo)
        fig.tight_layout()
        fig.savefig(caminho, dpi=120)
        plt.close(fig)
        print(f"  mapa salvo: {caminho}")


def executa(args):
    os.makedirs(args.saida, exist_ok=True)
    horas = list(args.janelas)

    # MERGE (observacao) - grade de referencia (10 km)
    merge_file = acha_arquivo_prev(args.base, MERGE_SUBDIR, "", MERGE_PADRAO)
    if merge_file is None:
        mfiles = sorted(glob.glob(os.path.join(args.base, MERGE_SUBDIR, "*.nc")))
        merge_file = mfiles[0] if mfiles else None
    if merge_file is None:
        print("ERRO: nenhum arquivo MERGE encontrado em "
              f"{os.path.join(args.base, MERGE_SUBDIR)}", file=sys.stderr)
        return 1
    print(f"MERGE (obs): {merge_file}")
    obs = MergeObs(merge_file)
    print(f"  grade MERGE: {obs.lats.size} x {obs.lons.size} "
          f"(verificacao nesta resolucao)")

    modelos = args.modelos or list(CONFIG.keys())
    linhas_par = []                      # serie por rodada/prazo
    acc_modelo, acc_lead = {}, {}        # acumuladores agregados
    mapas = {}

    def get_acc(dic, chave, tag):
        if chave not in dic:
            dic[chave] = dict(
                cont=AccContinuo(),
                cat=AccCategoria(args.limiares),
                fss=AccFSS(args.limiares, args.escalas_fss) if args.escalas_fss
                else None,
                tag=tag)
        return dic[chave]

    for modelo in modelos:
        cfg = CONFIG[modelo]
        runs = lista_rodadas(args.base, cfg["subdir"], args.max_rodadas)
        print(f"\n### Modelo '{modelo}' ({cfg['subdir']}): {len(runs)} rodadas")
        mapas[modelo] = Mapas(obs.grade)

        for ri, run in enumerate(runs, 1):
            arq = acha_arquivo_prev(args.base, cfg["subdir"], run, cfg["padrao"])
            if arq is None:
                print(f"  [{run}] arquivo '{cfg['padrao']}' nao encontrado")
                continue
            init = datetime.strptime(run, "%Y%m%d%H")
            try:
                janelas = previsao_acum24(arq, cfg, horas, init)
            except Exception as e:
                print(f"  [{run}] erro lendo previsao: {e}")
                continue

            for jw in janelas:
                if args.max_lead and jw.lead > args.max_lead:
                    continue
                campo_obs = obs.acum24(jw.fim)
                if campo_obs is None:
                    continue
                prev_rg = regrid(jw.campo, obs.grade,
                                 metodo=args.metodo_regrid)
                P, O = prev_rg.dados, campo_obs.dados

                # agregados
                for dic, chave, tag in [
                    (acc_modelo, (modelo, jw.hora),
                     dict(modelo=modelo, janela=f"{jw.hora:02d}Z")),
                    (acc_lead, (modelo, jw.hora, jw.lead),
                     dict(modelo=modelo, janela=f"{jw.hora:02d}Z", lead=jw.lead)),
                ]:
                    a = get_acc(dic, chave, tag)
                    a["cont"].add(P, O)
                    a["cat"].add(P, O)
                    if a["fss"]:
                        a["fss"].add(P, O)
                mapas[modelo].add(P, O)

                # serie por rodada/prazo (scores do par)
                sp = scores_par_continuo(P, O)
                linha = dict(modelo=modelo, rodada=run,
                             janela=f"{jw.hora:02d}Z", lead=jw.lead,
                             prazo_h=jw.prazo_h,
                             valido=jw.fim.strftime("%Y-%m-%d %HZ"), **sp)
                linhas_par.append(linha)
            if ri % 10 == 0 or ri == len(runs):
                print(f"  {ri}/{len(runs)} rodadas processadas")

    obs.close()

    # ---------------- saidas ----------------
    df_mod, df_mod_fss = _salva_placar(acc_modelo, os.path.join(args.saida, "placar_por_modelo.csv"))
    df_prz, df_prz_fss = _salva_placar(acc_lead, os.path.join(args.saida, "scores_por_prazo.csv"))
    df_rod = None
    if linhas_par:
        df_rod = pd.DataFrame(linhas_par)
        df_rod.to_csv(os.path.join(args.saida, "scores_por_rodada.csv"), index=False)
        print(f"\nscores_por_rodada.csv: {len(df_rod)} pares (rodada x prazo x janela)")
    else:
        print("\n[aviso] nenhum par previsao/observacao foi casado. "
              "Rode --inspecionar para conferir nomes/tempos.")

    if not args.sem_mapas:
        for modelo, mp in mapas.items():
            if mp.n.sum() > 0:
                mp.salva(os.path.join(args.saida, f"mapas_{modelo}.png"),
                         f"Verificacao 24h - modelo {modelo} (grade MERGE 10km)")

    if not args.sem_binario:
        binario = {
            "grade": {"lats": obs.lats, "lons": obs.lons},
            "config": {"janelas": horas, "limiares": list(args.limiares),
                       "escalas_fss": list(args.escalas_fss)},
            "mapas": {m: {"sp": mp.sp, "so": mp.so, "sdif": mp.sdif, "n": mp.n}
                      for m, mp in mapas.items()},
            "tabelas": {"placar_modelo": df_mod, "placar_modelo_fss": df_mod_fss,
                        "scores_prazo": df_prz, "scores_prazo_fss": df_prz_fss,
                        "scores_rodada": df_rod},
        }
        with open(os.path.join(args.saida, "verificacao.pkl"), "wb") as f:
            pickle.dump(binario, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("binario salvo: verificacao.pkl")

    print(f"\nResultados em: {os.path.abspath(args.saida)}")
    return 0


def _salva_placar(dic, caminho):
    """Consolida acumuladores em um CSV largo (continuas+categoricas+FSS)."""
    linhas = []
    for chave, a in sorted(dic.items()):
        base = dict(a["tag"])
        cont = a["cont"].scores()
        for cat in a["cat"].scores():
            linha = dict(base)
            linha.update({f"cont_{k}": v for k, v in cont.items()})
            linha.update(cat)
            linhas.append(linha)
    if not linhas:
        return None, None
    df = pd.DataFrame(linhas)
    df.to_csv(caminho, index=False)
    print(f"{os.path.basename(caminho)}: {len(df)} linhas")
    # FSS em arquivo separado (formato longo)
    fss_linhas = []
    for chave, a in sorted(dic.items()):
        if not a["fss"]:
            continue
        for r in a["fss"].scores():
            fss_linhas.append({**a["tag"], **r})
    df_fss = None
    if fss_linhas:
        df_fss = pd.DataFrame(fss_linhas)
        cam_fss = caminho.replace(".csv", "_fss.csv")
        df_fss.to_csv(cam_fss, index=False)
        print(f"{os.path.basename(cam_fss)}: {len(df_fss)} linhas")
    return df, df_fss


def parser():
    p = argparse.ArgumentParser(
        description="Verificacao de precip 24h do Eta (jaci/XC50) contra MERGE, "
                    "na resolucao do observado (10 km).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base", help="pasta raiz contendo nc/, nc_oper/, nc_merge/")
    p.add_argument("--saida", default="resultados_eta")
    p.add_argument("--modelos", nargs="*", choices=list(CONFIG.keys()),
                   default=None, help="quais modelos verificar (padrao: todos)")
    p.add_argument("--janelas", type=int, nargs="+", default=[12, 0],
                   help="horas finais das janelas de 24h (12=12Z-12Z, 0=00Z-00Z)")
    p.add_argument("--limiares", type=float, nargs="+", default=[0.254,2.54,6.35,12.7,19.05,25.4,38.1,50])
    p.add_argument("--escalas-fss", type=int, nargs="*", default=[1, 3, 5])
    p.add_argument("--max-lead", type=int, default=0,
                   help="prazo maximo em dias (0 = todos)")
    p.add_argument("--max-rodadas", type=int, default=0,
                   help="limita o n. de rodadas por modelo (0 = todas)")
    p.add_argument("--metodo-regrid", choices=["linear", "nearest"],
                   default="linear")
    p.add_argument("--sem-mapas", action="store_true")
    p.add_argument("--sem-binario", action="store_true")
    p.add_argument("--inspecionar", metavar="ARQ.nc", default=None,
                   help="apenas imprime a estrutura do arquivo e sai")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.inspecionar:
        inspeciona(args.inspecionar)
        return 0
    if not args.base:
        print("ERRO: informe --base (raiz com nc/, nc_oper/, nc_merge/) "
              "ou use --inspecionar ARQ.nc", file=sys.stderr)
        return 2
    try:
        return executa(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERRO: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
