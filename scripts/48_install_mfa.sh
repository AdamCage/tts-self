#!/usr/bin/env bash
# User-local MFA install (no sudo). Idempotent.
set -euo pipefail
export PATH="$HOME/bin:$HOME/micromamba/bin:$PATH"
mkdir -p "$HOME/bin" "$HOME/micromamba"
if ! command -v micromamba >/dev/null 2>&1; then
  echo "Installing micromamba to ~/bin ..."
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$HOME/bin" bin/micromamba
  mv "$HOME/bin/bin/micromamba" "$HOME/bin/micromamba" 2>/dev/null || true
  # tarball layout may be ./bin/micromamba
  if [[ -f $HOME/bin/bin/micromamba ]]; then mv "$HOME/bin/bin/micromamba" "$HOME/bin/micromamba"; fi
fi
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
if ! micromamba env list | grep -q '^mfa '; then
  micromamba create -y -n mfa -c conda-forge montreal-forced-aligner
fi
micromamba run -n mfa mfa version || true
micromamba run -n mfa mfa model download acoustic russian_mfa
micromamba run -n mfa mfa model download dictionary russian_mfa
# Convenience wrapper
cat > "$HOME/bin/mfa" <<'EOF'
#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
exec "$HOME/bin/micromamba" run -n mfa mfa "$@"
EOF
chmod +x "$HOME/bin/mfa"
echo "MFA wrapper: $HOME/bin/mfa"
"$HOME/bin/mfa" version
