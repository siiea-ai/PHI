import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import pytest

from phi.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "dashboard" / "pages"


# --------------------------- Helpers: CLI introspection --------------------------- #

def _get_cli_command(*names: str) -> click.Command:
    ctx = click.Context(cli_main)
    cmd: click.BaseCommand = cli_main
    for n in names:
        assert isinstance(cmd, click.Group), f"{cmd} is not a click.Group"
        cmd = cmd.get_command(ctx, n)  # type: ignore
        assert cmd is not None, f"Command not found: {' '.join(names)}"
    assert isinstance(cmd, click.Command)
    return cmd


def _cli_defaults_for(cmd: click.Command) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in cmd.params:
        if isinstance(p, click.Option):
            # Use parameter name (Python identifier), not flag string.
            name = p.name  # e.g., window_sec, phi_T0, phi_val
            out[name] = p.default
    return out


# --------------------------- Helpers: UI parsing --------------------------- #

_re_num_input = re.compile(r'st\.number_input\(\s*"(?P<name>[^"]+)"[\s\S]*?value=(?P<value>-?\d+(?:\.\d+)?)', re.MULTILINE)
_re_checkbox = re.compile(r'st\.checkbox\(\s*"(?P<name>[^"]+)"[\s\S]*?value=(?P<value>True|False)', re.MULTILINE)
_re_text_input_literal = re.compile(r'st\.text_input\(\s*"(?P<name>[^"]+)"[\s\S]*?value=\"(?P<value>[^\"]*)\"', re.MULTILINE)
_re_selectbox = re.compile(r'st\.selectbox\(\s*"(?P<name>[^"]+)"\s*,\s*options=\[(?P<options>[\s\S]*?)\]\s*,\s*index=(?P<index>\d+)', re.MULTILINE)
_re_multiselect = re.compile(r'st\.multiselect\(\s*"(?P<name>[^"]+)"\s*,\s*options=\[(?P<options>[\s\S]*?)\]\s*,\s*default=\[(?P<default>[\s\S]*?)\]', re.MULTILINE)

# Simple assignments like: const_v = 1.0; cos_period = 200; ...
_re_simple_assigns = {
    "const_v": re.compile(r'\bconst_v\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
    "cos_period": re.compile(r'\bcos_period\s*=\s*([0-9]+)'),
    "cos_min": re.compile(r'\bcos_min\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
    "cos_max": re.compile(r'\bcos_max\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
    "phi_T0": re.compile(r'\bphi_T0\s*=\s*([0-9]+)'),
    "phi": re.compile(r'\bphi\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
    "phi_min": re.compile(r'\bphi_min\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
    "phi_max": re.compile(r'\bphi_max\s*=\s*([0-9]+(?:\.[0-9]+)?)'),
}


def _parse_options_list(txt: str) -> List[str]:
    items = []
    for part in txt.split(','):
        part = part.strip()
        if part.startswith('"') and part.endswith('"'):
            items.append(part.strip('"'))
    return items


def _parse_list_literal(txt: str) -> List[str]:
    items = []
    for part in txt.split(','):
        part = part.strip()
        if part.startswith('"') and part.endswith('"'):
            items.append(part.strip('"'))
    return items


def _read(page: str) -> str:
    p = PAGES_DIR / page
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


# --------------------------- Page-specific parsers --------------------------- #

def parse_sim_ui_defaults() -> Dict[str, Any]:
    s = _read("01_BCI_Sim.py")
    out: Dict[str, Any] = {}
    # number_inputs
    for m in _re_num_input.finditer(s):
        name = m.group('name')
        val = float(m.group('value'))
        # integers should remain ints for steps/seed/periods
        if val.is_integer():
            val = int(val)
        out[name] = val
    # checkboxes
    for m in _re_checkbox.finditer(s):
        name = m.group('name')
        val = True if m.group('value') == 'True' else False
        out[name] = val
    # selectbox (scheduler default)
    ms = _re_selectbox.search(s)
    if ms:
        opts = _parse_options_list(ms.group('options'))
        idx = int(ms.group('index'))
        out[ms.group('name')] = opts[idx]
    # simple assigns for inactive scheduler params under default branch
    for key, rx in _re_simple_assigns.items():
        mm = rx.search(s)
        if mm and key not in out:
            val_s = mm.group(1)
            val = float(val_s)
            if val.is_integer():
                val = int(val)
            out[key] = val
    return out


def parse_sweep_ui_defaults() -> Dict[str, Any]:
    s = _read("02_BCI_Sweep.py")
    out: Dict[str, Any] = {}
    # number inputs
    for m in _re_num_input.finditer(s):
        name = m.group('name')
        val = float(m.group('value'))
        if val.is_integer():
            val = int(val)
        out[name] = val
    # text inputs with literal defaults
    for m in _re_text_input_literal.finditer(s):
        name = m.group('name')
        val = m.group('value')
        # ignore dynamically computed paths (value="" or non-literal handled elsewhere)
        out[name] = val
    # multiselect
    for m in _re_multiselect.finditer(s):
        name = m.group('name')
        defaults = _parse_list_literal(m.group('default'))
        out[name] = defaults
    # checkboxes
    for m in _re_checkbox.finditer(s):
        name = m.group('name')
        val = True if m.group('value') == 'True' else False
        out[name] = val
    return out


def parse_train_ui_defaults() -> Dict[str, Any]:
    s = _read("04_BCI_Train.py")
    out: Dict[str, Any] = {}
    # number inputs
    for m in _re_num_input.finditer(s):
        name = m.group('name')
        val = float(m.group('value'))
        if val.is_integer():
            val = int(val)
        out[name] = val
    # checkboxes
    for m in _re_checkbox.finditer(s):
        name = m.group('name')
        val = True if m.group('value') == 'True' else False
        out[name] = val
    # selectboxes
    for m in _re_selectbox.finditer(s):
        name = m.group('name')
        opts = _parse_options_list(m.group('options'))
        idx = int(m.group('index'))
        out[name] = opts[idx]
    # text inputs with literal (data_dir default is empty string)
    for m in _re_text_input_literal.finditer(s):
        out[m.group('name')] = m.group('value')
    return out


def parse_eval_ui_defaults() -> Dict[str, Any]:
    s = _read("03_BCI_Eval.py")
    out: Dict[str, Any] = {}
    # checkboxes
    for m in _re_checkbox.finditer(s):
        name = m.group('name')
        val = True if m.group('value') == 'True' else False
        out[name] = val
    # text inputs with literal default (out_path empty string)
    for m in _re_text_input_literal.finditer(s):
        out[m.group('name')] = m.group('value')
    return out


# --------------------------- Tests --------------------------- #

def test_parity_bci_sim_defaults():
    cmd = _get_cli_command("neuro", "bci-sim")
    cli_def = _cli_defaults_for(cmd)
    ui_def = parse_sim_ui_defaults()

    # Map UI keys to CLI param names where they differ
    key_map = {
        "phi": "phi_val",
    }

    keys_to_check = [
        "steps", "fs", "window_sec", "seed", "process_noise", "drift", "ctrl_effect",
        "base_lr", "base_gain", "scheduler", "const_v", "cos_period", "cos_min", "cos_max",
        "phi_T0", "phi", "phi_min", "phi_max", "save_features", "save_windows", "save_config",
    ]

    for k in keys_to_check:
        cli_k = key_map.get(k, k)
        assert cli_k in cli_def, f"Missing CLI default for {cli_k}"
        assert k in ui_def, f"Missing UI default for {k}"
        assert ui_def[k] == cli_def[cli_k], f"Mismatch {k}: UI={ui_def[k]} CLI={cli_def[cli_k]}"

    # Output dir behavior intentionally differs: CLI default None, UI pre-fills a dynamic path
    assert cli_def.get("out_dir", None) is None


def _as_list_of_ints(csv: str) -> List[int]:
    return [int(x) for x in csv.split(',') if x.strip()]


def _as_list_of_floats(csv: str) -> List[float]:
    return [float(x) for x in csv.split(',') if x.strip()]


def test_parity_bci_sweep_defaults():
    cmd = _get_cli_command("neuro", "bci-sweep")
    cli_def = _cli_defaults_for(cmd)
    ui_def = parse_sweep_ui_defaults()

    # Scalars
    for k in ["steps", "fs", "window_sec", "base_lr", "base_gain", "const_v", "cos_period", "cos_min", "cos_max", "phi_T0", "phi", "phi_min", "phi_max"]:
        ck = "phi_val" if k == "phi" else k
        assert ui_def[k] == cli_def[ck]

    # Multi lists
    assert _as_list_of_ints(ui_def["seeds (comma)"]) == list(cli_def["seeds"])  # type: ignore
    assert _as_list_of_floats(ui_def["process_noise (comma)"]) == list(cli_def["process_noise"])  # type: ignore
    assert _as_list_of_floats(ui_def["drift (comma)"]) == list(cli_def["drift"])  # type: ignore
    assert _as_list_of_floats(ui_def["snr_scale (comma)"]) == list(cli_def["snr_scale"])  # type: ignore
    assert _as_list_of_floats(ui_def["noise_std (comma)"]) == list(cli_def["noise_std"])  # type: ignore

    # Multiselect schedulers
    assert ui_def["schedulers"] == list(cli_def["schedulers"])  # type: ignore

    # Save flags
    assert ui_def["save_features"] == cli_def["save_features"]
    assert ui_def["save_windows"] == cli_def["save_windows"]
    assert ui_def["save_config"] == cli_def["save_config"]

    # Output root behavior differs intentionally
    assert "out_root" in cli_def and cli_def["out_root"] is None or cli_def["out_root"] == None  # required but no default in CLI


def test_parity_bci_train_defaults():
    cmd = _get_cli_command("neuro", "bci-train")
    cli_def = _cli_defaults_for(cmd)
    ui_def = parse_train_ui_defaults()

    assert ui_def["mode"] == cli_def["mode"]
    assert ui_def["model"] == cli_def["model"]
    assert ui_def["alpha (ridge/lasso)"] == cli_def["alpha"]
    assert ui_def["test_size"] == cli_def["test_size"]
    assert ui_def["random_state"] == cli_def["random_state"]
    assert ui_def["save_preds"] == cli_def["save_preds"]

    # data_dir is required in CLI; UI defaults to empty (session-filled). We don't compare.


def test_parity_bci_eval_defaults():
    cmd = _get_cli_command("neuro", "bci-eval")
    cli_def = _cli_defaults_for(cmd)
    ui_def = parse_eval_ui_defaults()

    # Flags
    assert ui_def["compute_signal_metrics"] == cli_def["compute_signal_metrics"]
    assert ui_def["save_plots"] == cli_def["save_plots"]

    # out_path: CLI default None vs UI empty string (not provided). We don't compare.
