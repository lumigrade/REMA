#!/bin/sh
# Install the "REMA Live" desktop app: launcher, icon, application entry, pinned to the GNOME Dash.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${REMA_DATA:-$HOME/rema-data}
mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/256x256/apps"
cp "$ROOT/docs/media/rema_icon.png" "$HOME/.local/share/icons/hicolor/256x256/apps/rema.png"
cat > "$HOME/.local/bin/rema-live" <<EOS
#!/bin/sh
exec "$DATA/envs/rig/bin/python" "$ROOT/tools/rema_app.py" "\$@"
EOS
chmod +x "$HOME/.local/bin/rema-live"
cat > "$HOME/.local/share/applications/rema.desktop" <<EOS
[Desktop Entry]
Type=Application
Name=REMA Live
GenericName=Virtual production compositor
Comment=Live actor matting over the Antilatency-tracked 3DGS scene (R record, I info, Q quit)
Exec=$HOME/.local/bin/rema-live
Icon=$HOME/.local/share/icons/hicolor/256x256/apps/rema.png
Terminal=false
StartupNotify=true
StartupWMClass=REMA
Categories=AudioVideo;Video;
EOS
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
if command -v gsettings >/dev/null; then
  cur=$(gsettings get org.gnome.shell favorite-apps)
  echo "$cur" | grep -q "'rema.desktop'" || gsettings set org.gnome.shell favorite-apps "$(echo "$cur" | sed "s/]\$/, 'rema.desktop']/")"
fi
echo "REMA Live installed and pinned to the Dash."
