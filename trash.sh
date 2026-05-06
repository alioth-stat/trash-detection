#!/usr/bin/env bash
# trash.sh — full workflow manager for the trash detection project
set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
TACO_DIR="$ROOT/data/raw/taco"
TRASHNET_DIR="$ROOT/data/raw/trashnet"
DETECTOR_DATA="$ROOT/data/processed/detector"
CLASSIFIER_DATA="$ROOT/data/processed/classifier"
WEIGHTS_DIR="$ROOT/weights"
DETECTOR_WEIGHTS="$WEIGHTS_DIR/detector.pt"
CLASSIFIER_WEIGHTS="$WEIGHTS_DIR/classifier.pt"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[•]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}━━  $*  ━━${RESET}"; }

die() { error "$*"; exit 1; }

# ── helpers ──────────────────────────────────────────────────────────────────
require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' not found. Install with: $2"
}

require_venv() {
    [[ -f "$PY" ]] || die "Virtual environment not found. Run:  $0 setup"
}

confirm() {
    read -rp "$(echo -e "${YELLOW}[?]${RESET} $* [y/N] ")" ans
    [[ "${ans,,}" == "y" ]]
}

# ── steps ────────────────────────────────────────────────────────────────────
cmd_setup() {
    header "Setup"
    require_cmd git   "sudo apt install git"
    require_cmd python3 "sudo apt install python3.12"

    if [[ ! -f "$PY" ]]; then
        info "Creating virtual environment at .venv ..."
        python3 -m venv "$VENV" || die "python3-venv missing. Run: sudo apt install python3.12-venv"
        success "Virtual environment created."
    else
        success "Virtual environment already exists."
    fi

    info "Installing / updating dependencies ..."
    "$PIP" install --upgrade pip --quiet
    "$PIP" install -r "$ROOT/requirements.txt" --quiet
    success "Dependencies installed."
}

cmd_download_taco() {
    header "Download TACO"
    require_venv

    if [[ ! -f "$TACO_DIR/download.py" ]]; then
        info "Cloning TACO repository ..."
        git clone https://github.com/pedropro/TACO.git "$TACO_DIR"
    else
        success "TACO repo already cloned."
    fi

    local ann="$TACO_DIR/data/annotations.json"
    [[ -f "$ann" ]] || die "Annotations not found at $ann"

    "$PIP" install requests Pillow --quiet

    info "Downloading images from Flickr (20-30 min, some 404s are normal) ..."
    cd "$TACO_DIR"
    "$PY" download.py --dataset_path "$ann" || warn "Download finished with errors — some images may be missing (normal)."
    cd "$ROOT"
    success "TACO download complete."
}

cmd_download_trashnet() {
    header "Download TrashNet"
    require_venv

    if [[ ! -d "$TRASHNET_DIR" ]]; then
        info "Cloning TrashNet repository ..."
        git clone https://github.com/garythung/trashnet.git "$TRASHNET_DIR"
    else
        success "TrashNet repo already cloned."
    fi

    if [[ -d "$TRASHNET_DIR/dataset-resized" ]]; then
        success "TrashNet images already extracted."
        return
    fi

    local archive="$TRASHNET_DIR/dataset-resized.7z"
    if [[ ! -f "$archive" ]]; then
        warn "TrashNet images require a manual download:"
        echo -e "  ${BOLD}1.${RESET} Open: https://github.com/garythung/trashnet/releases"
        echo -e "  ${BOLD}2.${RESET} Download dataset-resized.7z"
        echo -e "  ${BOLD}3.${RESET} Move it to: $archive"
        echo -e "  ${BOLD}4.${RESET} Re-run:  $0 download-trashnet"
        return
    fi

    require_cmd 7z "sudo apt install p7zip-full"
    info "Extracting dataset-resized.7z ..."
    7z x "$archive" -o"$TRASHNET_DIR" -y
    success "TrashNet extraction complete."
}

cmd_convert() {
    header "Convert Datasets"
    require_venv

    if [[ -n "$(find "$DETECTOR_DATA/images/train" -maxdepth 1 -name '*.jpg' -o -name '*.png' 2>/dev/null | head -1)" ]]; then
        success "Detector data already converted."
    else
        [[ -f "$TACO_DIR/data/annotations.json" ]] || die "TACO not downloaded yet. Run: $0 download-taco"
        info "Converting TACO → YOLO format ..."
        cd "$ROOT"
        "$PY" src/data/convert_taco.py "$TACO_DIR" --out "$DETECTOR_DATA"
        success "TACO converted."
    fi

    if [[ -n "$(find "$CLASSIFIER_DATA/train" -maxdepth 2 -name '*.jpg' -o -name '*.png' 2>/dev/null | head -1)" ]]; then
        success "Classifier data already converted."
    else
        [[ -d "$TRASHNET_DIR/dataset-resized" ]] || die "TrashNet not extracted yet. Run: $0 download-trashnet"
        info "Converting TrashNet → folder-per-class format ..."
        cd "$ROOT"
        "$PY" src/data/convert_trashnet.py "$TRASHNET_DIR" --out "$CLASSIFIER_DATA"
        success "TrashNet converted."
    fi
}

cmd_train_detector() {
    header "Train Detector (YOLOv8s)"
    require_venv
    [[ -d "$DETECTOR_DATA/images/train" ]] || die "Detector data missing. Run: $0 convert"

    info "Training YOLOv8s on TACO data (this will take a while on CPU) ..."
    cd "$ROOT"
    "$PY" train/train_detector.py --config configs/detector.yaml

    local best
    best=$(find "$ROOT/runs" -name "best.pt" -path "*/detector/*" | sort | tail -1)
    if [[ -n "$best" ]]; then
        mkdir -p "$WEIGHTS_DIR"
        cp "$best" "$DETECTOR_WEIGHTS"
        success "Detector weights saved to weights/detector.pt  (from $best)"
    else
        die "Training finished but best.pt not found anywhere under runs/"
    fi
}

cmd_train_classifier() {
    header "Train Classifier (EfficientNet-B0)"
    require_venv
    [[ -d "$CLASSIFIER_DATA/train" ]] || die "Classifier data missing. Run: $0 convert"

    info "Training EfficientNet-B0 on TrashNet data ..."
    cd "$ROOT"
    "$PY" train/train_classifier.py --config configs/classifier.yaml

    local best="$ROOT/runs/classifier/best.pt"
    if [[ -f "$best" ]]; then
        mkdir -p "$WEIGHTS_DIR"
        cp "$best" "$CLASSIFIER_WEIGHTS"
        success "Classifier weights saved to weights/classifier.pt"
    else
        die "Training finished but best.pt not found at runs/classifier/best.pt"
    fi
}

cmd_train() {
    cmd_train_detector
    cmd_train_classifier
}

cmd_run() {
    header "Run Live Detection"
    require_venv
    [[ -f "$DETECTOR_WEIGHTS" ]]   || die "Detector weights missing. Run: $0 train"
    [[ -f "$CLASSIFIER_WEIGHTS" ]] || die "Classifier weights missing. Run: $0 train"

    local camera="${1:-0}"
    local output="${2:-$ROOT/detections.jsonl}"

    info "Starting camera $camera  →  logging to $output"
    info "Press  q  in the video window to quit."
    cd "$ROOT"
    "$PY" detect.py --camera "$camera" --output "$output"
}

cmd_test() {
    header "Run Tests"
    require_venv
    cd "$ROOT"
    "$VENV/bin/pytest" tests/ -v
}

cmd_all() {
    header "Full Pipeline"
    cmd_setup
    cmd_download_taco
    cmd_download_trashnet
    cmd_convert
    cmd_train
    cmd_run
}

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    echo -e "${BOLD}Usage:${RESET}  $0 <command> [options]"
    echo
    echo -e "${BOLD}Commands:${RESET}"
    echo -e "  ${GREEN}setup${RESET}                Create venv and install dependencies"
    echo -e "  ${GREEN}download-taco${RESET}        Clone TACO repo and download images from Flickr"
    echo -e "  ${GREEN}download-trashnet${RESET}    Clone TrashNet repo (images need manual download)"
    echo -e "  ${GREEN}convert${RESET}              Convert both datasets to training format"
    echo -e "  ${GREEN}train${RESET}                Train detector + classifier"
    echo -e "  ${GREEN}train-detector${RESET}       Train YOLOv8s detector only"
    echo -e "  ${GREEN}train-classifier${RESET}     Train EfficientNet-B0 classifier only"
    echo -e "  ${GREEN}run${RESET} [cam] [out]      Run live camera detection"
    echo -e "                       cam: camera index (default 0)"
    echo -e "                       out: output jsonl path (default detections.jsonl)"
    echo -e "  ${GREEN}test${RESET}                 Run pytest suite"
    echo -e "  ${GREEN}all${RESET}                  Run entire pipeline from scratch"
    echo
    echo -e "${BOLD}Examples:${RESET}"
    echo -e "  $0 setup"
    echo -e "  $0 download-taco"
    echo -e "  $0 convert"
    echo -e "  $0 train"
    echo -e "  $0 run"
    echo -e "  $0 run 1 my_session.jsonl"
}

# ── dispatch ──────────────────────────────────────────────────────────────────
case "${1:-}" in
    setup)               cmd_setup ;;
    download-taco)       cmd_download_taco ;;
    download-trashnet)   cmd_download_trashnet ;;
    convert)             cmd_convert ;;
    train)               cmd_train ;;
    train-detector)      cmd_train_detector ;;
    train-classifier)    cmd_train_classifier ;;
    run)                 cmd_run "${2:-0}" "${3:-$ROOT/detections.jsonl}" ;;
    test)                cmd_test ;;
    all)                 cmd_all ;;
    ""|--help|-h)        usage ;;
    *)                   error "Unknown command: $1"; usage; exit 1 ;;
esac
