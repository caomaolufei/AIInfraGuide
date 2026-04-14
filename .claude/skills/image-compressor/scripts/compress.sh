#!/bin/bash
# Image compression script
# Usage: compress.sh <input> [--quality Q] [--resize W] [--format fmt] [--output path]
# Examples:
#   compress.sh photo.png                          # Auto-compress PNG
#   compress.sh photo.png --quality 40             # pngquant quality 0-40
#   compress.sh photo.png --format jpg --quality 30 --resize 1024  # Convert to JPEG, resize, compress
#   compress.sh photo.jpg --quality 50             # JPEG recompress

set -e

INPUT=""
QUALITY=""
RESIZE=""
FORMAT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quality|-q) QUALITY="$2"; shift 2 ;;
    --resize|-r)  RESIZE="$2"; shift 2 ;;
    --format|-f)  FORMAT="$2"; shift 2 ;;
    --output|-o)  OUTPUT="$2"; shift 2 ;;
    *) INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "ERROR: Input file not found: $INPUT" >&2
  exit 1
fi

DIR="$(dirname "$INPUT")"
BASE="$(basename "$INPUT")"
NAME="${BASE%.*}"
EXT="${BASE##*.}"
EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"

# Detect source format
SRC_FMT="$EXT_LOWER"
[[ "$SRC_FMT" == "jpg" ]] && SRC_FMT="jpeg"

# Determine target format
if [[ -n "$FORMAT" ]]; then
  TGT_FMT="$(echo "$FORMAT" | tr '[:upper:]' '[:lower:]')"
  [[ "$TGT_FMT" == "jpg" ]] && TGT_FMT="jpeg"
else
  TGT_FMT="$SRC_FMT"
fi

# Set target extension
case "$TGT_FMT" in
  jpeg) TGT_EXT="jpg" ;;
  png)  TGT_EXT="png" ;;
  *)    TGT_EXT="$TGT_FMT" ;;
esac

# Set output path
if [[ -z "$OUTPUT" ]]; then
  if [[ "$TGT_EXT" == "$EXT_LOWER" ]]; then
    OUTPUT="${DIR}/${NAME}_compressed.${TGT_EXT}"
  else
    OUTPUT="${DIR}/${NAME}.${TGT_EXT}"
  fi
fi

ORIG_SIZE=$(stat -f%z "$INPUT" 2>/dev/null || stat -c%s "$INPUT" 2>/dev/null)

# --- PNG -> PNG (pngquant) ---
if [[ "$SRC_FMT" == "png" && "$TGT_FMT" == "png" ]]; then
  Q="${QUALITY:-40}"
  if ! command -v pngquant &>/dev/null; then
    echo "Installing pngquant..." >&2
    if command -v brew &>/dev/null; then
      brew install pngquant >&2
    elif command -v apt-get &>/dev/null; then
      sudo apt-get install -y pngquant >&2
    else
      echo "ERROR: pngquant not found. Install it manually." >&2
      exit 1
    fi
  fi
  TMPFILE="$(mktemp "${DIR}/${NAME}_tmp_XXXX.png")"
  # Resize first if needed
  if [[ -n "$RESIZE" ]]; then
    sips -Z "$RESIZE" "$INPUT" --out "$TMPFILE" >/dev/null 2>&1
    pngquant --quality=0-"$Q" --speed 1 --force --output "$OUTPUT" "$TMPFILE"
    rm -f "$TMPFILE"
  else
    rm -f "$TMPFILE"
    pngquant --quality=0-"$Q" --speed 1 --force --output "$OUTPUT" "$INPUT"
  fi

# --- Any -> JPEG (sips) ---
elif [[ "$TGT_FMT" == "jpeg" ]]; then
  Q="${QUALITY:-40}"
  if [[ -n "$RESIZE" ]]; then
    TMPFILE="$(mktemp "${DIR}/${NAME}_tmp_XXXX.png")"
    sips -Z "$RESIZE" "$INPUT" --out "$TMPFILE" >/dev/null 2>&1
    sips -s format jpeg -s formatOptions "$Q" "$TMPFILE" --out "$OUTPUT" >/dev/null 2>&1
    rm -f "$TMPFILE"
  else
    sips -s format jpeg -s formatOptions "$Q" "$INPUT" --out "$OUTPUT" >/dev/null 2>&1
  fi

# --- JPEG -> JPEG (recompress) ---
elif [[ "$SRC_FMT" == "jpeg" && "$TGT_FMT" == "jpeg" ]]; then
  Q="${QUALITY:-40}"
  if [[ -n "$RESIZE" ]]; then
    TMPFILE="$(mktemp "${DIR}/${NAME}_tmp_XXXX.jpg")"
    sips -Z "$RESIZE" "$INPUT" --out "$TMPFILE" >/dev/null 2>&1
    sips -s format jpeg -s formatOptions "$Q" "$TMPFILE" --out "$OUTPUT" >/dev/null 2>&1
    rm -f "$TMPFILE"
  else
    sips -s format jpeg -s formatOptions "$Q" "$INPUT" --out "$OUTPUT" >/dev/null 2>&1
  fi

# --- Fallback: resize only via sips ---
else
  if [[ -n "$RESIZE" ]]; then
    sips -Z "$RESIZE" "$INPUT" --out "$OUTPUT" >/dev/null 2>&1
  else
    cp "$INPUT" "$OUTPUT"
    echo "WARN: No compression applied for $SRC_FMT -> $TGT_FMT" >&2
  fi
fi

NEW_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
RATIO=$(echo "scale=1; (1 - $NEW_SIZE / $ORIG_SIZE) * 100" | bc)

echo "INPUT:  $INPUT ($(echo "scale=0; $ORIG_SIZE / 1024" | bc)KB)"
echo "OUTPUT: $OUTPUT ($(echo "scale=0; $NEW_SIZE / 1024" | bc)KB)"
echo "RATIO:  ${RATIO}%"
