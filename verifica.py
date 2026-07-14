#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verifica.py — driver unificado da verificacao do modelo Eta.

Le um unico arquivo de parametros (config_verificacao.yaml), calcula as metricas
de cada componente (precipitacao x MERGE; campos continuos x ERA5) e grava:
  - <saida>/verificacao.pkl : binario com TUDO para regerar figuras;
  - <saida>/<componente>_continuas.csv, _categoricas.csv, _fss.csv : dados das figuras.

Depois, use gera_figuras.py para (re)plotar sem reprocessar os dados raw.

Uso:
  python verifica.py --config config_verificacao.yaml
  python verifica.py --config config_verificacao.yaml --componentes precipitacao t2m
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

import nucleo as N

# motor de precipitacao (acum24 + categoricas + fss) ja testado
try:
    import precip_engine as PE
except ImportError:
    import verifica_eta_deliv as PE   # nome usado nos testes locais

# leitor binctl (opcional; so se algum modelo usar reader: binctl)
try:
    import leitor_binctl as BINCTL
except Exception:
    BINCTL = None


# ==========================================================================
# REFERENCIAS
# ==========================================================================
class RefMerge:
    """Observacao MERGE (acum 24h) — reaproveita o MergeObs do motor de precip."""
    def __init__(self, arquivo):
        self.obs = PE.MergeObs(arquivo)
        self.grade = self.obs.grade
        self.lats = self.obs.lats; self.lons = self.obs.lons

    def campo(self, valido, **_):
        return self.obs.acum24(valido)

    def close(self):
        self.obs.close()


class RefEra5:
    """Reanalise ERA5 (arquivo unico multi-variavel, campos instantaneos)."""
    def __init__(self, arquivo):
        self.ds = N.abre(arquivo)
        _, self.latn, self.lonn, self.tn = N.nomes(self.ds, prefer="t2m|msl|u10")
        self.lats, self.lons = N.grade(self.ds, self.latn, self.lonn)
        self.tempos = (pd.to_datetime(self.ds[self.tn].values)
                       if self.tn and self.tn in self.ds else None)
        self.grade = N.Campo(np.zeros((self.lats.size, self.lons.size)),
                             self.lats, self.lons, nome="era5")

    def campo(self, valido, var=None, nivel=None, unidade=None, para=None):
        da = self.ds[var]
        if self.tempos is not None:
            dif = np.abs((self.tempos - pd.Timestamp(valido)).total_seconds())
            i = int(np.argmin(dif))
            if dif[i] > 6 * 3600:
                return None
        else:
            i = 0
        arr = N.campo_2d(da, self.latn, self.lonn, self.tn, i, nivel=nivel)
        if unidade and para:
            arr = N.converte_unidade(arr, unidade, para)
        return N.Campo(arr, self.lats, self.lons, nome=var)

    def campo_mag(self, valido, uvar, vvar, unidade=None, para=None):
        """Magnitude do vento sqrt(u^2+v^2) da reanalise no tempo valido."""
        cu = self.campo(valido, var=uvar, unidade=unidade, para=para)
        cv = self.campo(valido, var=vvar, unidade=unidade, para=para)
        if cu is None or cv is None:
            return None
        mag = np.sqrt(cu.dados ** 2 + cv.dados ** 2)
        return N.Campo(mag, self.lats, self.lons, nome="mag")

    def close(self):
        self.ds.close()


# ==========================================================================
# LEITURA DE CAMPOS DO MODELO (instantaneos)
# ==========================================================================
def _lista_rodadas(base, subdir, glob_init="*", limite=0):
    d = os.path.join(base, subdir)
    if not os.path.isdir(d):
        return []
    runs = sorted(x for x in os.listdir(d)
                  if re.fullmatch(r"\d{10}", x)
                  and os.path.isdir(os.path.join(d, x)))
    if glob_init and glob_init != "*":
        import fnmatch
        runs = [r for r in runs if fnmatch.fnmatch(r, glob_init)]
    return runs[:limite] if limite else runs


def _acha_arq(base, subdir, run, padrao):
    d = os.path.join(base, subdir, run)
    achados = sorted(glob.glob(os.path.join(d, padrao)))
    if achados:
        return achados[0]
    # fallback case-insensitive (ex.: TP2M_*.nc casa tp2m_*.nc)
    import fnmatch
    if os.path.isdir(d):
        pl = padrao.lower()
        cand = sorted(f for f in os.listdir(d) if fnmatch.fnmatch(f.lower(), pl))
        if cand:
            return os.path.join(d, cand[0])
    return None


def _resolve_arq(cfg, mspec, modelo, run, reader, padrao=None):
    """Caminho do arquivo do modelo para a rodada.
    - netcdf: <base>/<subdir>/<run>/<padrao>  (glob)
    - binctl: <binctl_base>/<run>/<member>/<ctl com {init}>  (GrADS .ctl)"""
    if reader == "binctl":
        base = mspec.get("binctl_base") or cfg.get("binctl_base") or cfg["base"]
        member = mspec.get("member", "")
        ctlname = mspec["ctl"].format(init=run)
        p = (os.path.join(base, run, member, ctlname) if member
             else os.path.join(base, run, ctlname))
        return p if os.path.exists(p) else None
    subdir = mspec.get("subdir") or cfg["modelos"][modelo]["subdir"]
    return _acha_arq(cfg["base"], subdir, run, padrao)


def campos_instantaneos(arq, spec, leads_horas, init, reader="netcdf"):
    """Retorna lista de (valido, lead_h, lead_d, Campo) do modelo."""
    if reader == "binctl":
        if BINCTL is None:
            raise RuntimeError("reader=binctl mas leitor_binctl indisponivel")
        return BINCTL.campos_instantaneos(arq, spec, leads_horas, init)
    # NetCDF
    ds = N.abre(arq)
    var, latn, lonn, tn = N.nomes(ds, spec.get("var"),
                                  prefer="tp2m|t2m|pslm|psnm|u10|v10|zgeo|z")
    lats, lons = N.grade(ds, latn, lonn)
    da = ds[var]
    tempos = (pd.to_datetime(ds[tn].values) if tn and tn in ds else None)
    unidade = spec.get("unidade"); para = spec.get("para")
    nivel = spec.get("nivel")
    saida = []
    for lh in leads_horas:
        valido = pd.Timestamp(init) + timedelta(hours=int(lh))
        if tempos is not None:
            dif = np.abs((tempos - valido).total_seconds())
            i = int(np.argmin(dif))
            if dif[i] > 90 * 60:
                continue
        else:
            i = 0
        arr = N.campo_2d(da, latn, lonn, tn, i, nivel=nivel)
        if unidade and para:
            arr = N.converte_unidade(arr, unidade, para)
        ld = int(round(lh / 24.0))
        saida.append((valido, int(lh), ld, N.Campo(arr, lats, lons, nome=var)))
    ds.close()
    return saida


# ==========================================================================
# MAPAS ESPACIAIS: media diaria por (modelo, prazo, mes)
# ==========================================================================
def _quer_mapa(cfg, lead):
    mc = cfg.get("mapas", {}) or {}
    if not mc.get("ativar", True):
        return False
    leads = mc.get("leads")
    return (leads is None) or (lead in leads)


def _rotulo_mes(cfg, valido):
    mc = cfg.get("mapas", {}) or {}
    if not mc.get("por_mes", True):
        return 0                       # 0 = agrega todo o periodo
    return int(pd.Timestamp(valido).month)


def _meses_ok(cfg):
    """Meses aceitos nos mapas (default Jan/Fev/Mar). 0 = agregado sempre entra."""
    mc = cfg.get("mapas", {}) or {}
    return set(mc.get("meses", [1, 2, 3])) | {0}


def _periodo_ok(cfg, valido):
    """Filtro GLOBAL de meses (afeta TODAS as metricas: curvas + mapas).
    cfg['meses'] = lista de meses a verificar (default Jan/Fev/Mar). Vazio/None
    = sem filtro (usa todo o periodo dos dados)."""
    ms = cfg.get("meses", [1, 2, 3])
    if not ms:
        return True
    return int(pd.Timestamp(valido).month) in set(ms)


def _acumula_mapa(cfg, mapas, ref, modelo, lead, valido, P, O):
    if not _quer_mapa(cfg, lead):
        return
    mes = _rotulo_mes(cfg, valido)
    if mes not in _meses_ok(cfg):
        return
    mapas.setdefault((modelo, lead, mes), N.Mapas(ref.grade)).add(P, O)


# ==========================================================================
# COMPONENTE: PRECIPITACAO (acum24 + regioes + categoricas/fss)
# ==========================================================================
def _run_precip(cfg, comp, ref, masks, regioes, modelo, run,
                cont, cat, fss, mapas, max_lead):
    mspec = comp["modelos"][modelo]
    horas = cfg.get("janelas_precip", [12, 0])
    limiares = comp.get("limiares", [0.254, 2.54, 6.35, 12.7, 19.05, 25.4, 38.1, 50])
    escalas = comp.get("escalas_fss", [1, 3, 5, 11])
    quer_cat = "categoricas" in comp.get("metricas", [])
    quer_fss = "fss" in comp.get("metricas", [])
    pe_cfg = {"subdir": cfg["modelos"][modelo]["subdir"], "padrao": mspec["padrao"],
              "var": mspec.get("var"), "fator": mspec.get("fator_para_mm", 1.0),
              "acumulado": mspec.get("acumulado", False)}
    arq = _acha_arq(cfg["base"], pe_cfg["subdir"], run, pe_cfg["padrao"])
    if arq is None:
        return
    init = datetime.strptime(run, "%Y%m%d%H")
    try:
        janelas = PE.previsao_acum24(arq, pe_cfg, horas, init)
    except Exception as e:
        print(f"    [{run}] erro: {e}"); return
    for jw in janelas:
        if max_lead and jw.lead > max_lead:
            continue
        if not _periodo_ok(cfg, jw.fim):        # so meses completos (Jan/Fev/Mar)
            continue
        co = ref.campo(jw.fim)
        if co is None:
            continue
        P = PE.regrid(jw.campo, ref.grade, metodo=cfg.get("regrid", "linear")).dados
        O = co.dados
        for rg in regioes:
            mk = masks[rg]
            cont.setdefault((modelo, rg, jw.lead), N.AccCont()).add_caso(P[mk], O[mk])
        if quer_cat:
            cat.setdefault((modelo, jw.lead), N.AccCategoria(limiares)).add(P, O)
        if quer_fss:
            fss.setdefault((modelo, jw.lead), N.AccFSS(limiares, escalas)).add(P, O)
        _acumula_mapa(cfg, mapas, ref, modelo, jw.lead, jw.fim, P, O)


def _campos_magnitude(cfg, comp, modelo, run, leads_horas, init):
    """Le u e v do modelo e devolve a magnitude sqrt(u^2+v^2) por tempo valido."""
    mspec = comp["modelos"][modelo]
    subdir = mspec.get("subdir", cfg["modelos"][modelo]["subdir"])
    reader = mspec.get("reader", cfg["modelos"][modelo].get("reader", "netcdf"))
    uu = mspec["u"]; vv = mspec["v"]; para = comp.get("unidade")
    uspec = {"var": uu.get("var"), "unidade": mspec.get("unidade"), "para": para}
    vspec = {"var": vv.get("var"), "unidade": mspec.get("unidade"), "para": para}
    if reader == "binctl":                      # u e v no MESMO .ctl
        uarq = varq = _resolve_arq(cfg, mspec, modelo, run, "binctl")
    else:
        uarq = _resolve_arq(cfg, mspec, modelo, run, reader, uu["padrao"])
        varq = _resolve_arq(cfg, mspec, modelo, run, reader, vv["padrao"])
    if uarq is None or varq is None:
        return []
    uc = campos_instantaneos(uarq, uspec, leads_horas, init, reader)
    vc = campos_instantaneos(varq, vspec, leads_horas, init, reader)
    vmap = {c[0]: c[3] for c in vc}
    saida = []
    for valido, lh, ld, ucampo in uc:
        vcampo = vmap.get(valido)
        if vcampo is None:
            continue
        mag = np.sqrt(ucampo.dados ** 2 + vcampo.dados ** 2)
        saida.append((valido, lh, ld, N.Campo(mag, ucampo.lats, ucampo.lons, "mag")))
    return saida


def _run_instant(cfg, comp, ref, masks, regioes, modelo, run,
                 cont, cat, fss, mapas, max_lead):
    mspec = comp["modelos"][modelo]
    leads_horas = cfg.get("leads_horas", [24, 48, 72, 96, 120, 144, 168, 192, 216])
    e5 = comp.get("era5", {})
    init = datetime.strptime(run, "%Y%m%d%H")
    magnitude = comp.get("derivada") == "magnitude"
    try:
        if magnitude:
            campos = _campos_magnitude(cfg, comp, modelo, run, leads_horas, init)
        else:
            reader = mspec.get("reader", cfg["modelos"][modelo].get("reader", "netcdf"))
            spec = {"var": mspec.get("var"), "unidade": mspec.get("unidade"),
                    "para": comp.get("unidade"), "nivel": mspec.get("nivel")}
            arq = _resolve_arq(cfg, mspec, modelo, run, reader, mspec.get("padrao"))
            campos = campos_instantaneos(arq, spec, leads_horas, init, reader) if arq else []
    except Exception as ex:
        print(f"    [{run}] erro: {ex}"); return
    for valido, lh, ld, campo in campos:
        if max_lead and ld > max_lead:
            continue
        if not _periodo_ok(cfg, valido):        # so meses completos (Jan/Fev/Mar)
            continue
        if magnitude:
            co = ref.campo_mag(valido, e5.get("u"), e5.get("v"),
                               unidade=e5.get("unidade"), para=comp.get("unidade"))
        else:
            co = ref.campo(valido, var=e5.get("var"), nivel=e5.get("nivel"),
                           unidade=e5.get("unidade"), para=comp.get("unidade"))
        if co is None:
            continue
        P = N.regrid(campo, ref.grade, metodo=cfg.get("regrid", "linear")).dados
        O = co.dados
        for rg in regioes:
            mk = masks[rg]
            cont.setdefault((modelo, rg, ld), N.AccCont()).add_caso(P[mk], O[mk])
        _acumula_mapa(cfg, mapas, ref, modelo, ld, valido, P, O)


def _acumula_run(cfg, comp, ref, masks, regioes, modelo, run, max_lead,
                 cont, cat, fss, mapas):
    fn = _run_precip if comp["tipo"] == "acum24" else _run_instant
    fn(cfg, comp, ref, masks, regioes, modelo, run, cont, cat, fss, mapas, max_lead)


def _abre_ref(cfg, nome):
    rc = cfg["referencias"][nome]
    if nome == "merge":
        arq = _acha_arq(cfg["base"], rc["subdir"], "", rc["padrao"]) or \
            sorted(glob.glob(os.path.join(cfg["base"], rc["subdir"], "*.nc")))[0]
        return RefMerge(arq)
    if nome == "era5":
        return RefEra5(rc["arquivo"])
    raise ValueError(f"referencia desconhecida: {nome}")


def _regioes_de(ref, cfg):
    masks, tem = N.constroi_mascaras(ref.lats, ref.lons,
                                     cfg.get("regioes", {}).get("caixas_br"))
    regd = ["Todo"] + (["Continente", "Oceano"] if tem else [])
    return masks, regd, regd + list(N.CAIXAS_BR.keys())


def _tarefas(cfg, comp, max_rodadas):
    T = []
    for modelo, mspec in comp["modelos"].items():
        reader = mspec.get("reader",
                           cfg["modelos"].get(modelo, {}).get("reader", "netcdf"))
        if reader == "binctl":
            base = mspec.get("binctl_base") or cfg.get("binctl_base") or cfg["base"]
            runs = _lista_rodadas(base, "", cfg.get("init_glob", "*"), max_rodadas)
        else:
            subdir = mspec.get("subdir") or cfg["modelos"][modelo]["subdir"]
            runs = _lista_rodadas(cfg["base"], subdir, cfg.get("init_glob", "*"),
                                  max_rodadas)
        for run in runs:
            T.append((modelo, run))
    return T


def _worker(payload):
    """Processa um subconjunto de (modelo, run) num processo separado.
    Cada worker abre a sua propria referencia e devolve acumuladores parciais."""
    cfg, comp, tarefas, max_lead = payload
    ref = _abre_ref(cfg, comp["referencia"])
    masks, _regd, regioes = _regioes_de(ref, cfg)
    cont = {}; cat = {}; fss = {}; mapas = {}
    for modelo, run in tarefas:
        _acumula_run(cfg, comp, ref, masks, regioes, modelo, run, max_lead,
                     cont, cat, fss, mapas)
    ref.close()
    return cont, cat, fss, mapas


def _merge_dict(dst, src):
    for k, v in src.items():
        if k in dst:
            dst[k].merge(v)
        else:
            dst[k] = v


def roda_componente(cfg, comp, ref, masks, regioes, args):
    tarefas = _tarefas(cfg, comp, args.max_rodadas)
    print(f"  [{comp['nome']}] {len(tarefas)} rodadas-modelo")
    cont = {}; cat = {}; fss = {}; mapas = {}
    if getattr(args, "jobs", 1) and args.jobs > 1 and len(tarefas) > 1:
        import multiprocessing as mp
        nj = min(args.jobs, len(tarefas))
        chunks = [tarefas[i::nj] for i in range(nj)]      # round-robin (balanceado)
        payloads = [(cfg, comp, ch, args.max_lead) for ch in chunks]
        print(f"    paralelo: {nj} processos")
        with mp.Pool(nj) as pool:
            for pc, pk, pf, pm in pool.map(_worker, payloads):
                _merge_dict(cont, pc); _merge_dict(cat, pk)
                _merge_dict(fss, pf); _merge_dict(mapas, pm)
    else:
        for modelo, run in tarefas:
            _acumula_run(cfg, comp, ref, masks, regioes, modelo, run,
                         args.max_lead, cont, cat, fss, mapas)
    return _empacota(comp, cont, cat, fss, mapas, ref)


def _empacota(comp, cont, cat, fssacc, mapas, ref):
    linhas = [dict(modelo=mo, regiao=rg, lead=ld, **a.scores())
              for (mo, rg, ld), a in cont.items()]
    dfc = pd.DataFrame(linhas)
    dfk = pd.DataFrame([dict(modelo=mo, lead=ld, **r)
                        for (mo, ld), a in cat.items() for r in a.scores()])
    dff = pd.DataFrame([dict(modelo=mo, lead=ld, **r)
                        for (mo, ld), a in fssacc.items() for r in a.scores()])
    # mapas keyed por (modelo, lead, mes); arrays em float32 p/ binario menor
    mp = {k: {"sp": v.sp.astype("float32"), "so": v.so.astype("float32"),
              "n": v.n.astype("float32")} for k, v in mapas.items()}
    return dict(nome=comp["nome"], unidade=comp.get("unidade", ""),
                tipo=comp["tipo"], continuas=dfc, categoricas=dfk, fss=dff,
                mapas=mp, grade={"lats": ref.lats, "lons": ref.lons})


# ==========================================================================
# MAIN
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Driver unificado de verificacao Eta.")
    ap.add_argument("--config", default="config_verificacao.yaml")
    ap.add_argument("--componentes", nargs="*", default=None,
                    help="subconjunto de componentes a processar")
    ap.add_argument("--max-rodadas", type=int, default=0)
    ap.add_argument("--max-lead", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="numero de processos paralelos (divide as rodadas). "
                         "0 ou negativo = usa todos os cores disponiveis.")
    args = ap.parse_args(argv)
    if args.jobs is not None and args.jobs <= 0:
        import multiprocessing as mp
        args.jobs = mp.cpu_count()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    saida = cfg.get("saida", "resultados_uni")
    os.makedirs(saida, exist_ok=True)

    # referencias (uma instancia por tipo, reutilizada)
    refs = {}

    def get_ref(nome):
        if nome not in refs:
            refs[nome] = _abre_ref(cfg, nome)
        return refs[nome]

    componentes = cfg["componentes"]
    escolhidos = args.componentes or list(componentes.keys())
    resultado = {}
    for nome in escolhidos:
        comp = dict(componentes[nome]); comp["nome"] = nome
        print(f"\n=== Componente: {nome} ({comp['tipo']}, ref={comp['referencia']}) ===")
        ref = get_ref(comp["referencia"])
        masks, regd, regioes = _regioes_de(ref, cfg)
        resultado[nome] = roda_componente(cfg, comp, ref, masks, regioes, args)
        resultado[nome]["regioes_dominio"] = regd
        # aviso claro quando o componente nao casou nenhum par
        dfc = resultado[nome].get("continuas")
        npares = 0 if (dfc is None or dfc.empty) else int(dfc["n"].sum())
        if npares == 0:
            print(f"  *** ATENCAO: componente '{nome}' NAO casou nenhum par. "
                  f"Confira em config: padrao/var dos arquivos do modelo, o "
                  f"arquivo/variaveis do ERA5, leads_horas e o reader. ***")
        else:
            print(f"  {nome}: {npares} pontos-caso casados; "
                  f"mapas: {len(resultado[nome].get('mapas', {}))}")

        # CSVs por componente
        r = resultado[nome]
        for suf, df in [("continuas", r["continuas"]), ("categoricas", r["categoricas"]),
                        ("fss", r["fss"])]:
            if df is not None and not df.empty:
                df.to_csv(os.path.join(saida, f"{nome}_{suf}.csv"), index=False)

    for r in refs.values():
        r.close()

    with open(os.path.join(saida, "verificacao.pkl"), "wb") as f:
        pickle.dump(resultado, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nBinario e CSVs em: {os.path.abspath(saida)}")
    print("Gere as figuras com: python gera_figuras.py --binario "
          f"{os.path.join(saida, 'verificacao.pkl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
