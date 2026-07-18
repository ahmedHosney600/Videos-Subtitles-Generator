"""
convert_arabic_model.py
========================
Converts Byne/whisper-large-v3-arabic (PyTorch/HuggingFace format)
to MLX format so it can be used natively with mlx-whisper on Apple Silicon.

Output: ./models/whisper-large-v3-arabic-mlx/
  ├── weights.safetensors  (MLX weights)
  └── config.json          (MLX-compatible config)

Usage:
    source .venv/bin/activate
    python3 convert_arabic_model.py
"""

import json
import sys
from pathlib import Path

# ── Rich console for nice output ──────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    console = Console()
except ImportError:
    class Console:
        def print(self, *a, **kw): print(*a)
        def status(self, *a, **kw):
            import contextlib
            return contextlib.nullcontext()
    console = Console()

HF_REPO    = "Byne/whisper-large-v3-arabic"
OUTPUT_DIR = Path("models/whisper-large-v3-arabic-mlx")

# Whisper Large-v3 architecture dimensions (same base as openai/whisper-large-v3)
WHISPER_LARGE_V3_DIMS = {
    "n_mels": 128,
    "n_audio_ctx": 1500,
    "n_audio_state": 1280,
    "n_audio_head": 20,
    "n_audio_layer": 32,
    "n_vocab": 51866,
    "n_text_ctx": 448,
    "n_text_state": 1280,
    "n_text_head": 20,
    "n_text_layer": 32,
}


def check_dependencies():
    missing = []
    for pkg in ["mlx", "torch", "transformers", "huggingface_hub", "safetensors"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        console.print(f"[red]✗ Missing packages: {', '.join(missing)}[/]")
        console.print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    console.print("  [green]✓[/] All dependencies present")


def download_model():
    """Download the PyTorch model from HuggingFace."""
    from huggingface_hub import snapshot_download
    console.print(f"\n  [cyan]Downloading[/] [bold]{HF_REPO}[/] from HuggingFace…")
    console.print("  [dim](~3 GB — this will take a few minutes)[/]\n")

    local_path = snapshot_download(
        repo_id=HF_REPO,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    console.print(f"  [green]✓[/] Downloaded to: {local_path}")
    return Path(local_path)


def load_pytorch_weights(model_path: Path) -> dict:
    """Load weights from PyTorch checkpoint or safetensors."""
    import torch

    # Try safetensors first (preferred)
    st_files = list(model_path.glob("model.safetensors")) + \
               list(model_path.glob("*.safetensors"))
    pt_files = list(model_path.glob("pytorch_model.bin")) + \
               list(model_path.glob("model.bin"))

    if st_files:
        console.print(f"  [dim]Loading safetensors: {st_files[0].name}[/]")
        from safetensors.torch import load_file
        return load_file(st_files[0])
    elif pt_files:
        console.print(f"  [dim]Loading PyTorch bin: {pt_files[0].name}[/]")
        return torch.load(pt_files[0], map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_path}\n"
            f"Files: {list(model_path.iterdir())}"
        )


# ── PyTorch → MLX key mapping ─────────────────────────────────────────────────
# HuggingFace Whisper uses different key names than OpenAI/MLX Whisper.
# This mapping translates HF key patterns → MLX key patterns.

HF_TO_MLX_KEY_MAP = {
    # Encoder
    "model.encoder.conv1.weight":              "encoder.conv1.weight",
    "model.encoder.conv1.bias":                "encoder.conv1.bias",
    "model.encoder.conv2.weight":              "encoder.conv2.weight",
    "model.encoder.conv2.bias":                "encoder.conv2.bias",
    "model.encoder.embed_positions.weight":    "encoder.positional_embedding",
    # Decoder
    "model.decoder.embed_tokens.weight":       "decoder.token_embedding.weight",
    "model.decoder.embed_positions.weight":    "decoder.positional_embedding",
    # Final layer norms
    "model.encoder.layer_norm.weight":         "encoder.ln_post.weight",
    "model.encoder.layer_norm.bias":           "encoder.ln_post.bias",
    "model.decoder.layer_norm.weight":         "decoder.ln.weight",
    "model.decoder.layer_norm.bias":           "decoder.ln.bias",
    # proj_out (logits)
    "proj_out.weight":                         "decoder.token_embedding.weight",  # tied
}


def remap_key(key: str) -> str | None:
    """Map a HuggingFace Whisper key to an MLX Whisper key. Returns None to skip."""
    # Direct map
    if key in HF_TO_MLX_KEY_MAP:
        return HF_TO_MLX_KEY_MAP[key]

    # Skip proj_out (weight-tied with token embedding)
    if key == "proj_out.weight":
        return None

    # Encoder layers: model.encoder.layers.N.XXX → encoder.blocks.N.XXX
    if key.startswith("model.encoder.layers."):
        rest = key[len("model.encoder.layers."):]
        return remap_layer_key(rest, "encoder.blocks")

    # Decoder layers: model.decoder.layers.N.XXX → decoder.blocks.N.XXX
    if key.startswith("model.decoder.layers."):
        rest = key[len("model.decoder.layers."):]
        return remap_layer_key(rest, "decoder.blocks")

    # Pass through anything not matched (shouldn't happen for clean Whisper checkpoints)
    return key


def remap_layer_key(rest: str, prefix: str) -> str:
    """
    Remap sub-keys within a transformer block.
    rest format: "N.sub_module.param"  (e.g. "0.self_attn.q_proj.weight")
    """
    # Split layer index from the rest
    parts = rest.split(".", 1)
    layer_idx = parts[0]
    sub = parts[1] if len(parts) > 1 else ""

    # Attention sub-module mappings
    attn_map = {
        "self_attn.q_proj":       "attn.query",
        "self_attn.k_proj":       "attn.key",
        "self_attn.v_proj":       "attn.value",
        "self_attn.out_proj":     "attn.out",
        "self_attn_layer_norm":   "attn_ln",
        "encoder_attn.q_proj":    "cross_attn.query",
        "encoder_attn.k_proj":    "cross_attn.key",
        "encoder_attn.v_proj":    "cross_attn.value",
        "encoder_attn.out_proj":  "cross_attn.out",
        "encoder_attn_layer_norm":"cross_attn_ln",
        "fc1":                    "mlp.0",
        "fc2":                    "mlp.2",
        "final_layer_norm":       "mlp_ln",
        "self_attn.k_proj":       "attn.key",
    }

    for hf_sub, mlx_sub in attn_map.items():
        if sub.startswith(hf_sub):
            param = sub[len(hf_sub):]  # e.g. ".weight"
            return f"{prefix}.{layer_idx}.{mlx_sub}{param}"

    # Fallback: keep as-is under the new prefix
    return f"{prefix}.{layer_idx}.{sub}"


def convert_weights(pt_weights: dict) -> dict:
    """Convert a PyTorch weight dict to MLX format."""
    import torch
    import mlx.core as mx

    mlx_weights = {}
    skipped = []

    for pt_key, pt_tensor in pt_weights.items():
        mlx_key = remap_key(pt_key)

        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # Convert: PyTorch tensor → numpy → MLX array
        np_array = pt_tensor.float().numpy()
        mlx_weights[mlx_key] = mx.array(np_array)

    if skipped:
        console.print(f"  [dim]Skipped {len(skipped)} weight-tied keys (normal)[/]")

    console.print(f"  [green]✓[/] Converted {len(mlx_weights)} weight tensors")
    return mlx_weights


def save_mlx_model(mlx_weights: dict, output_dir: Path):
    """Save MLX weights and config to disk."""
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = str(output_dir / "weights.safetensors")
    console.print(f"  [dim]Saving weights → {weights_path}[/]")
    mx.save_safetensors(weights_path, mlx_weights)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(WHISPER_LARGE_V3_DIMS, f, indent=2)

    console.print(f"  [green]✓[/] Saved config → {config_path}")


def verify_model(output_dir: Path):
    """Quick sanity check: load the converted model with mlx-whisper."""
    console.print("\n  [cyan]Verifying converted model loads correctly…[/]")
    try:
        from mlx_whisper.load_models import load_model
        model = load_model(str(output_dir))
        console.print("  [green]✓[/] Model loads successfully with mlx-whisper!")
        del model
    except Exception as e:
        console.print(f"  [yellow]⚠  Verification warning: {e}[/]")
        console.print("  [dim]The model may still work — try it with transcribe.py[/]")


def main():
    import subprocess
    
    if sys.platform == "darwin":
        console.print("\n[bold cyan]━━━ Whisper Arabic → MLX Converter ━━━[/]")
        console.print(f"  Source : [bold]{HF_REPO}[/]")
        console.print(f"  Output : [bold]{OUTPUT_DIR}[/]\n")

        # 1. Check deps
        console.print("[bold]Step 1/5[/] Checking dependencies…")
        check_dependencies()

        # 2. Check if already converted
        if (OUTPUT_DIR / "weights.safetensors").exists():
            console.print(f"\n[yellow]⚠  Model already converted at {OUTPUT_DIR}[/]")
            console.print("  Delete the folder and re-run to reconvert.")
            console.print("\n[green]✓ Nothing to do — model is ready![/]")
            return

        # 3. Download
        console.print("\n[bold]Step 2/5[/] Downloading from HuggingFace…")
        model_path = download_model()

        # 4. Load PyTorch weights
        console.print("\n[bold]Step 3/5[/] Loading PyTorch weights…")
        pt_weights = load_pytorch_weights(model_path)
        console.print(f"  [green]✓[/] Loaded {len(pt_weights)} weight tensors")

        # 5. Convert to MLX
        console.print("\n[bold]Step 4/5[/] Converting PyTorch → MLX format…")
        mlx_weights = convert_weights(pt_weights)

        # 6. Save
        console.print("\n[bold]Step 5/5[/] Saving MLX model…")
        save_mlx_model(mlx_weights, OUTPUT_DIR)

        # 7. Verify
        verify_model(OUTPUT_DIR)

        console.print(f"""
[bold green]━━━ Conversion Complete! ━━━[/]

  Model saved to: [bold]{OUTPUT_DIR.resolve()}[/]

  To use it, run:
    [cyan]python3 transcribe.py[/]
  Then select [bold]Arabic Fine-tuned[/] in the speed mode menu.
""")
    else:
        CT2_OUTPUT = Path("models/whisper-large-v3-arabic-ct2")
        console.print("\n[bold cyan]━━━ Whisper Arabic → CTranslate2 Converter ━━━[/]")
        console.print(f"  Source : [bold]{HF_REPO}[/]")
        console.print(f"  Output : [bold]{CT2_OUTPUT}[/]\n")
        
        if (CT2_OUTPUT / "model.bin").exists():
            console.print(f"\n[yellow]⚠  Model already converted at {CT2_OUTPUT}[/]")
            return
            
        console.print("\n[bold]Step 1/1[/] Running ct2-transformers-converter…")
        try:
            subprocess.run([
                "ct2-transformers-converter", 
                "--model", HF_REPO,
                "--output_dir", str(CT2_OUTPUT),
                "--copy_files", "tokenizer.json", "preprocessor_config.json",
                "--quantization", "float16"
            ], check=True)
            console.print(f"\n[bold green]━━━ Conversion Complete! ━━━[/]")
        except Exception as e:
            console.print(f"\n[bold red]✗ Conversion failed: {e}[/]")


if __name__ == "__main__":
    main()
