#!/bin/bash
# ==========================================================================
# commit.sh - adiciona, commita, integra o remoto e envia para o GitHub.
#
# Uso:
#   ./commit.sh                       # mensagem automatica com data/hora
#   ./commit.sh "sua mensagem aqui"   # mensagem personalizada
#
# Primeira vez (repo ainda nao existe no GitHub), com o gh CLI autenticado:
#   git init && gh repo create Verificacao --public --source=. --push
#
# (na primeira execucao: chmod +x commit.sh)
# ==========================================================================
set -e
cd "$(dirname "$0")"

# 1) precisa ser um repositorio git
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERRO: esta pasta nao e' um repositorio git."
    echo "Inicialize (com o gh CLI autenticado):"
    echo "  git init && gh repo create Verificacao --public --source=. --push"
    exit 1
fi

# 2) mensagem
MSG="$*"
if [ -z "$MSG" ]; then
    MSG="atualizacao $(date '+%Y-%m-%d %H:%M')"
fi

# 3) stage + commit (se houver algo)
git add -A
if git diff --cached --quiet; then
    echo "Nada novo para commitar."
else
    echo "Arquivos no commit:"
    git diff --cached --name-status
    git commit -m "$MSG"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)

# 4) se ja existe upstream, integra o remoto ANTES de empurrar (evita rejeicao)
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
    echo "Integrando o remoto (merge) antes do push..."
    git config pull.rebase false
    if ! git pull --no-edit; then
        echo ""
        echo "*** Conflito no merge. Resolva os arquivos marcados, depois:"
        echo "      git add <arquivos> && git commit --no-edit && ./commit.sh"
        exit 1
    fi
    git push
else
    # primeiro push deste ramo
    git push -u origin "$BRANCH"
fi

echo "OK: '$MSG' enviado para origin/$BRANCH."
