#!/bin/bash
# ==========================================================================
# setup_env.sh - cria (ou atualiza) o ambiente conda 'verificacao' numa area
# GRAVAVEL (importante em clusters onde o Anaconda base e' compartilhado e o
# pkgs dir padrao ~/conda/pkgs nao e' gravavel).
#
#   ./setup_env.sh
#   # (opcional) apontar outra area gravavel:
#   CONDA_STORE=/caminho/gravavel/conda ./setup_env.sh
#
# Depois:  conda activate verificacao
# ==========================================================================
set -e
cd "$(dirname "$0")"

ENVNAME=verificacao

# Area conda gravavel (pkgs + envs). No ian01/grpeta, use o diretorio do projeto.
CONDA_STORE=${CONDA_STORE:-/p/projetos/grpeta/Team/jorge.gomes/conda}
export CONDA_PKGS_DIRS="$CONDA_STORE/pkgs"
ENVPREFIX="$CONDA_STORE/envs/$ENVNAME"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_STORE/envs"

# usa mamba se disponivel (bem mais rapido), senao conda
CONDA=conda
command -v mamba >/dev/null 2>&1 && CONDA=mamba

echo ">> pkgs dir: $CONDA_PKGS_DIRS"
echo ">> env prefix: $ENVPREFIX"

if [ -d "$ENVPREFIX" ]; then
    echo ">> Ambiente ja existe: atualizando..."
    $CONDA env update -p "$ENVPREFIX" -f environment.yml --prune
else
    echo ">> Criando ambiente..."
    $CONDA env create -p "$ENVPREFIX" -f environment.yml
fi

echo ""
echo "Pronto. Ative com (nome ou prefixo):"
echo "    conda activate ${ENVNAME}"
echo "    # ou: conda activate ${ENVPREFIX}"
echo "Teste rapido:"
echo "    python verifica.py --config config_verificacao.yaml --max-rodadas 1"
