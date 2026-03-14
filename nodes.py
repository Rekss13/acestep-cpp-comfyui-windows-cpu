import os
import json
import platform
import multiprocessing
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import folder_paths

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def load_config() -> Dict[str, Any]:
    """Load config from config.json; return empty dict if missing or invalid."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_user_model_folders() -> List[str]:
    """Return user-specified model folders from config.json."""
    return load_config().get("model_folders", [])


def get_merged_model_folders() -> List[str]:
    """Merge ComfyUI text_encoders folders with any user-configured folders."""
    try:
        comfy_folders = folder_paths.get_folder_paths("text_encoders")
    except Exception:
        comfy_folders = []
    all_folders = comfy_folders + get_user_model_folders()
    return [f for f in all_folders if os.path.isdir(f)]


def scan_gguf_models() -> List[str]:
    """Scan registered 'acestep_gguf' folder type for .gguf files.

    Tries ComfyUI's folder_paths registry first (so the built-in model
    manager can track the files), then falls back to manual scanning.
    """
    try:
        all_files = folder_paths.get_filename_list("acestep_gguf")
        gguf_files = sorted(f for f in all_files if f.lower().endswith(".gguf"))
        if gguf_files:
            return gguf_files
    except Exception:
        pass

    # Fallback: manual scan of text_encoders + user-configured folders
    seen_models: set = set()
    models: List[str] = []
    for folder in get_merged_model_folders():
        try:
            for name in os.listdir(folder):
                if name.lower().endswith(".gguf") and name not in seen_models:
                    models.append(name)
                    seen_models.add(name)
        except OSError:
            pass
    return sorted(models)


def find_model_path(model_name: str) -> Optional[str]:
    """Return the full path to a model file, or None if not found."""
    for folder in get_merged_model_folders():
        path = os.path.join(folder, model_name)
        if os.path.isfile(path):
            return path
    return None


def _coerce_float(value: Any, default: float) -> float:
    """Return *value* as a float, falling back to *default* for empty strings.

    Older ComfyUI workflows may store optional FLOAT widget values as ``""``
    instead of the numeric default when the field was not explicitly set.
    """
    if isinstance(value, str):
        return float(value) if value.strip() else default
    return float(value)


def _coerce_int(value: Any, default: int) -> int:
    """Return *value* as an int, falling back to *default* for empty strings.

    Older ComfyUI workflows may store optional INT widget values as ``""``
    instead of the numeric default when the field was not explicitly set.
    """
    if isinstance(value, str):
        return int(value) if value.strip() else default
    return int(value)
def _binary_in_build(build_dir: str, name: str) -> Optional[str]:
    """Return the path to *name* if it exists in *build_dir* or *build_dir*/bin.

    ggml's CMakeLists.txt sets ``CMAKE_RUNTIME_OUTPUT_DIRECTORY`` to
    ``${CMAKE_BINARY_DIR}/bin`` when it is used as a cmake subdirectory (the
    normal case for acestep.cpp).  The cmake configure command now passes this
    variable explicitly, but we still check *build_dir*/bin/ so that existing
    installations built before this fix are found without a forced rebuild.
    """
    for candidate in (
        os.path.join(build_dir, name),
        os.path.join(build_dir, "bin", name),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def get_binary_path(binary_name: str) -> Optional[str]:
    """
    Locate an acestep.cpp binary.

    Search order:
      1. Explicit path from config.json ``binary_paths`` mapping.
      2. System PATH (via shutil.which).
      3. ``<node_dir>/acestep.cpp/build/<binary_name>`` (local build alongside the node).
      4. ``<node_dir>/acestep.cpp/build/bin/<binary_name>`` (ggml's default output
         directory when used as a cmake subdirectory).
    """
    config = load_config()
    explicit = config.get("binary_paths", {}).get(binary_name)
    if explicit and os.path.isfile(explicit):
        return explicit

    on_path = shutil.which(binary_name)
    if on_path:
        return on_path

    build_base = os.path.join(os.path.dirname(__file__), "acestep.cpp", "build")
    return _binary_in_build(build_base, binary_name)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# HuggingFace repo that hosts all pre-quantized ACE-Step GGUFs
_HF_REPO = "Serveurperso/ACE-Step-1.5-GGUF"

# Quant resolution rules mirroring models.sh
_EMB_QUANTS = ["Q8_0", "BF16"]
_LM_SMALL_QUANTS = ["Q8_0", "BF16"]
_LM_4B_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "BF16"]
_DIT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "BF16"]
_DIT_VARIANTS = ["turbo", "sft", "base", "turbo-shift1", "turbo-shift3", "turbo-continuous"]
_LM_SIZES = ["4B", "1.7B", "0.6B"]


def _resolve_quant(requested: str, model_type: str) -> str:
    """Mirror the quant resolution logic from models.sh."""
    if model_type in ("emb", "lm_small"):
        return requested if requested == "BF16" else "Q8_0"
    if model_type == "lm_4B":
        if requested in ("Q4_K_M", "Q5_K_M"):
            return "Q5_K_M"
        return requested if requested in ("BF16", "Q8_0", "Q6_K") else "Q8_0"
    return requested  # dit: all quants available


class AcestepCPPModelDownloader:
    """
    Download ACE-Step GGUF models from HuggingFace (``Serveurperso/ACE-Step-1.5-GGUF``)
    directly into a local folder so they appear in the Model Loader dropdown.
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            default_dir = folder_paths.get_folder_paths("text_encoders")[0]
        except Exception:
            default_dir = os.path.join(os.path.dirname(__file__), "models")

        return {
            "required": {
                "save_dir": (
                    "STRING",
                    {
                        "default": default_dir,
                        "tooltip": "Directory to save downloaded GGUF files into",
                    },
                ),
                "lm_size": (
                    _LM_SIZES,
                    {"default": "4B", "tooltip": "LM model size"},
                ),
                "quant": (
                    _DIT_QUANTS,
                    {
                        "default": "Q8_0",
                        "tooltip": (
                            "Quantisation level. "
                            "Embedding/LM models fall back to the nearest available quant "
                            "automatically (mirroring models.sh logic)."
                        ),
                    },
                ),
                "dit_variant": (
                    _DIT_VARIANTS,
                    {"default": "turbo", "tooltip": "DiT model variant to download"},
                ),
            },
            "optional": {
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "HuggingFace access token (leave empty for public repos)",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Re-download even if the file already exists",
                        "label_on": "Overwrite",
                        "label_off": "Skip existing",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("downloaded_files",)
    FUNCTION = "download"
    CATEGORY = "AcestepCPP"
    OUTPUT_NODE = True

    def download(
        self,
        save_dir: str,
        lm_size: str = "4B",
        quant: str = "Q8_0",
        dit_variant: str = "turbo",
        hf_token: str = "",
        overwrite: bool = False,
    ):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloading. "
                "Install it with: pip install huggingface_hub"
            )

        os.makedirs(save_dir, exist_ok=True)
        token = hf_token.strip() or None

        lm_type = "lm_4B" if lm_size == "4B" else "lm_small"
        files_to_download = [
            "vae-BF16.gguf",
            f"Qwen3-Embedding-0.6B-{_resolve_quant(quant, 'emb')}.gguf",
            f"acestep-5Hz-lm-{lm_size}-{_resolve_quant(quant, lm_type)}.gguf",
            f"acestep-v15-{dit_variant}-{_resolve_quant(quant, 'dit')}.gguf",
        ]

        downloaded: List[str] = []
        skipped: List[str] = []

        for filename in files_to_download:
            dest = os.path.join(save_dir, filename)
            if os.path.isfile(dest) and not overwrite:
                print(f"[AcestepCPP] Skip (exists): {filename}")
                skipped.append(filename)
                continue

            print(f"[AcestepCPP] Downloading: {filename} ...")
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=filename,
                local_dir=save_dir,
                token=token,
            )
            print(f"[AcestepCPP] Done: {filename}")
            downloaded.append(filename)

        summary_parts = []
        if downloaded:
            summary_parts.append(f"Downloaded: {', '.join(downloaded)}")
        if skipped:
            summary_parts.append(f"Skipped (already exist): {', '.join(skipped)}")
        summary = "\n".join(summary_parts) if summary_parts else "Nothing to do."
        return (summary,)


# The upstream source repository for the binaries
_ACESTEP_CPP_REPO = "https://github.com/audiohacking/acestep.cpp"


class AcestepCPPBuilder:
    """
    Clone ``acestep.cpp`` from GitHub and build the ``ace-qwen3`` and
    ``dit-vae`` binaries via CMake.

    The default clone directory (``<node_dir>/acestep.cpp``) is the same path
    that ``get_binary_path()`` already searches, so the built binaries will be
    found automatically by the Generate node without any extra configuration.
    """

    BACKENDS = ["auto", "cuda", "metal", "blas", "cpu"]

    @classmethod
    def INPUT_TYPES(cls):
        default_dir = os.path.join(os.path.dirname(__file__), "acestep.cpp")
        return {
            "required": {
                "clone_dir": (
                    "STRING",
                    {
                        "default": default_dir,
                        "tooltip": (
                            "Directory to clone acestep.cpp into. "
                            "Defaults to <node_dir>/acestep.cpp so that the "
                            "Generate node finds the binaries automatically."
                        ),
                    },
                ),
                "backend": (
                    cls.BACKENDS,
                    {
                        "default": "auto",
                        "tooltip": (
                            "GPU/compute backend for CMake. "
                            "'auto' detects CUDA, Metal, or falls back to CPU."
                        ),
                    },
                ),
            },
            "optional": {
                "force_rebuild": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Delete the existing build directory and rebuild from scratch."
                        ),
                        "label_on": "Force rebuild",
                        "label_off": "Incremental",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("build_log",)
    FUNCTION = "build"
    CATEGORY = "AcestepCPP"
    OUTPUT_NODE = True

    # Maximum number of recent log lines included in error messages
    _MAX_ERROR_LOG_LINES = 40

    # Binaries produced by the cmake build
    _BINARIES = ("ace-qwen3", "dit-vae")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        """Return the best available cmake backend flag value."""
        # CUDA: look for nvcc or nvidia-smi on PATH
        if shutil.which("nvcc") or shutil.which("nvidia-smi"):
            return "cuda"
        # Apple Metal: macOS always supports Metal through ggml
        if platform.system() == "Darwin":
            return "metal"
        # OpenBLAS: check pkg-config first (cross-distro), then common header
        # locations, then dpkg as a last resort on Debian-based systems
        if shutil.which("pkg-config") and subprocess.run(
            ["pkg-config", "--exists", "openblas"],
            capture_output=True,
        ).returncode == 0:
            return "blas"
        openblas_headers = [
            "/usr/include/openblas/cblas.h",
            "/usr/local/include/openblas/cblas.h",
            "/opt/homebrew/include/openblas/cblas.h",
        ]
        if any(os.path.isfile(h) for h in openblas_headers):
            return "blas"
        if shutil.which("dpkg") and subprocess.run(
            ["dpkg", "-l", "libopenblas-dev"],
            capture_output=True,
        ).returncode == 0:
            return "blas"
        return "cpu"

    @staticmethod
    def _cmake_flags(backend: str) -> List[str]:
        """Translate backend name to CMake -D flags."""
        mapping = {
            "cuda":  ["-DGGML_CUDA=ON"],
            "metal": [],  # Metal is auto-enabled on macOS by ggml
            "blas":  ["-DGGML_BLAS=ON"],
            "cpu":   [],
        }
        return mapping.get(backend, [])

    @staticmethod
    def _run(cmd: List[str], cwd: str, log_lines: List[str]) -> None:
        """Run a command, stream output to log_lines, raise on failure."""
        label = " ".join(cmd)
        log_lines.append(f"$ {label}")
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True
        )
        if result.stdout:
            log_lines.extend(result.stdout.splitlines())
        if result.stderr:
            log_lines.extend(result.stderr.splitlines())
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): {label}\n"
                + "\n".join(log_lines[-AcestepCPPBuilder._MAX_ERROR_LOG_LINES:])
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        clone_dir: str,
        backend: str = "auto",
        force_rebuild: bool = False,
    ):
        log: List[str] = []

        # --- Resolve backend --------------------------------------------------
        resolved = self._detect_backend() if backend == "auto" else backend
        log.append(f"[AcestepCPP] Backend: {resolved} (requested: {backend})")

        # --- Clone or update --------------------------------------------------
        repo_dir = clone_dir
        if not os.path.isdir(repo_dir):
            log.append(f"[AcestepCPP] Cloning {_ACESTEP_CPP_REPO} -> {repo_dir}")
            if not shutil.which("git"):
                raise RuntimeError(
                    "git is not available on PATH. Please install git and retry."
                )
            self._run(
                ["git", "clone", "--recurse-submodules", _ACESTEP_CPP_REPO, repo_dir],
                cwd=os.path.dirname(repo_dir),
                log_lines=log,
            )
        else:
            log.append(f"[AcestepCPP] Repo exists: {repo_dir} — updating submodules")
            self._run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=repo_dir,
                log_lines=log,
            )

        # --- Prepare build directory ------------------------------------------
        build_dir = os.path.join(repo_dir, "build")
        if force_rebuild and os.path.isdir(build_dir):
            log.append(f"[AcestepCPP] Removing existing build dir: {build_dir}")
            shutil.rmtree(build_dir)

        os.makedirs(build_dir, exist_ok=True)

        # --- CMake configure --------------------------------------------------
        # Pass CMAKE_RUNTIME_OUTPUT_DIRECTORY explicitly so that ggml's
        # CMakeLists.txt (which defaults to ${CMAKE_BINARY_DIR}/bin when used
        # as a subdirectory) does not redirect ace-qwen3 and dit-vae into
        # build/bin/ instead of build/.
        cmake_cmd = [
            "cmake", "..",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_dir}",
        ] + self._cmake_flags(resolved)
        log.append(f"[AcestepCPP] Configuring: {' '.join(cmake_cmd)}")
        if not shutil.which("cmake"):
            raise RuntimeError(
                "cmake is not available on PATH. "
                "Install cmake (e.g. apt install cmake) and retry."
            )
        self._run(cmake_cmd, cwd=build_dir, log_lines=log)

        # --- CMake build ------------------------------------------------------
        jobs = str(multiprocessing.cpu_count())
        build_cmd = ["cmake", "--build", ".", "--config", "Release", f"-j{jobs}"]
        log.append(f"[AcestepCPP] Building with {jobs} parallel jobs ...")
        self._run(build_cmd, cwd=build_dir, log_lines=log)

        # --- Verify outputs ---------------------------------------------------
        # _binary_in_build checks both build/ (new cmake configure with explicit
        # output dir) and build/bin/ (ggml's default when
        # CMAKE_RUNTIME_OUTPUT_DIRECTORY is not set).
        missing = [
            binary for binary in self._BINARIES
            if _binary_in_build(build_dir, binary) is None
        ]

        if missing:
            raise RuntimeError(
                f"Build succeeded but expected binaries not found: {', '.join(missing)}. "
                f"Check the build log above."
            )

        log.append(
            f"[AcestepCPP] Build complete. Binaries in {build_dir}: "
            + ", ".join(self._BINARIES)
        )
        return ("\n".join(log),)


class AcestepCPPLoraLoader:
    """
    Specify a LoRA adapter file for use with acestep.cpp.

    Enter the full path to any ``.gguf`` or ``.safetensors`` LoRA file on
    your filesystem. Outputs an ``ACESTEP_LORA`` dict consumed by the
    generator node.
    """

    _ALLOWED_EXTENSIONS = (".gguf", ".safetensors")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Full path to a LoRA adapter file (.gguf or .safetensors). "
                            "You can use any file location on your filesystem."
                        ),
                    },
                ),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "LoRA adapter scale",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ACESTEP_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "AcestepCPP"

    def load_lora(self, lora_path: str, lora_scale: float):
        path = lora_path.strip()
        if not path:
            raise ValueError(
                "lora_path is empty. Enter the full path to your LoRA file."
            )
        if not any(path.lower().endswith(ext) for ext in self._ALLOWED_EXTENSIONS):
            raise ValueError(
                f"Unsupported LoRA file type: {path!r}. "
                f"Expected one of: {', '.join(self._ALLOWED_EXTENSIONS)}"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LoRA file not found: {path}"
            )
        return ({"path": path, "scale": lora_scale},)


class AcestepCPPOptions:
    """
    Advanced technical options for acestep.cpp generation.

    Configures output format, VAE memory tiling, batch counts, and debug
    flags that are typically set once and reused across multiple generations.
    Connect the output to the **Acestep.cpp Generate** node's ``options``
    input.  All parameters are optional — unset fields fall back to their
    acestep.cpp defaults.
    """

    OUTPUT_FORMATS = ["mp3", "wav"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                # Output format
                "output_format": (
                    cls.OUTPUT_FORMATS,
                    {
                        "default": "mp3",
                        "tooltip": "Audio output format: MP3 (smaller) or WAV (lossless)",
                    },
                ),
                "mp3_bitrate": (
                    "INT",
                    {
                        "default": 128,
                        "min": 64,
                        "max": 320,
                        "tooltip": "MP3 bitrate in kbps (only used when output_format is 'mp3')",
                    },
                ),
                # VAE memory control
                "vae_chunk": (
                    "INT",
                    {
                        "default": 256,
                        "min": 16,
                        "max": 1024,
                        "tooltip": (
                            "VAE latent frames per tile (default: 256). "
                            "Reduce to lower VRAM usage at the cost of speed."
                        ),
                    },
                ),
                "vae_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 256,
                        "tooltip": "VAE overlap frames per side (default: 64)",
                    },
                ),
                # Batch generation
                "lm_batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 9,
                        "tooltip": (
                            "Number of LM (ace-qwen3) sequences to generate in parallel. "
                            "Each element produces a genuinely different song from a different seed."
                        ),
                    },
                ),
                "dit_batch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 9,
                        "tooltip": (
                            "Number of DiT (dit-vae) variations per LM output (max 9). "
                            "Variations share the same prompt but differ in initial noise."
                        ),
                    },
                ),
                # Advanced / debug
                "no_flash_attn": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Disable flash attention in both ace-qwen3 and dit-vae",
                        "label_on": "Disabled",
                        "label_off": "Enabled",
                    },
                ),
                "lm_max_seq": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 1024,
                        "max": 65536,
                        "tooltip": "ace-qwen3 KV cache size in tokens (default: 8192)",
                    },
                ),
                "lm_no_fsm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Disable FSM constrained decoding in ace-qwen3",
                        "label_on": "Disabled",
                        "label_off": "Enabled",
                    },
                ),
            }
        }

    RETURN_TYPES = ("ACESTEP_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "get_options"
    CATEGORY = "AcestepCPP"

    def get_options(self, **kwargs) -> Tuple:
        return (dict(kwargs),)


class AcestepCPPModelLoader:
    """
    Select the four GGUF model files required by acestep.cpp.

    Outputs an ``ACESTEP_MODELS`` dict consumed by the generator node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_list = scan_gguf_models()
        options = model_list if model_list else ["No GGUF models found"]
        return {
            "required": {
                "lm_model": (
                    options,
                    {
                        "tooltip": (
                            "LM (ace-qwen3) model GGUF, e.g. "
                            "acestep-5Hz-lm-4B-Q8_0.gguf"
                        )
                    },
                ),
                "text_encoder_model": (
                    options,
                    {
                        "tooltip": (
                            "Text-encoder GGUF, e.g. "
                            "Qwen3-Embedding-0.6B-Q8_0.gguf"
                        )
                    },
                ),
                "dit_model": (
                    options,
                    {
                        "tooltip": (
                            "DiT GGUF, e.g. "
                            "acestep-v15-turbo-Q8_0.gguf"
                        )
                    },
                ),
                "vae_model": (
                    options,
                    {"tooltip": "VAE GGUF, e.g. vae-BF16.gguf"},
                ),
            }
        }

    RETURN_TYPES = ("ACESTEP_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_models"
    CATEGORY = "AcestepCPP"

    def load_models(
        self,
        lm_model: str,
        text_encoder_model: str,
        dit_model: str,
        vae_model: str,
    ):
        paths = {
            "lm_model": find_model_path(lm_model),
            "text_encoder": find_model_path(text_encoder_model),
            "dit": find_model_path(dit_model),
            "vae": find_model_path(vae_model),
        }

        missing = [
            label
            for label, path in [
                ("LM model", paths["lm_model"]),
                ("text encoder", paths["text_encoder"]),
                ("DiT model", paths["dit"]),
                ("VAE model", paths["vae"]),
            ]
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Could not locate model file(s): {', '.join(missing)}. "
                "Check your model folder configuration."
            )

        return (paths,)


class AcestepCPPGenerate:
    """
    Generate music with acestep.cpp.

    Runs ``ace-qwen3`` (LM) then ``dit-vae`` (DiT + VAE).  The native MP3 or
    WAV produced by the binary is copied as-is to ComfyUI's output directory
    and an inline audio player is displayed on the node — no Python audio
    decoding or re-encoding is performed.  Connect an **AcestepCPPOptions**
    node to the ``options`` input to control output format, batching, VAE
    tiling, and advanced debug flags.
    """

    VOCAL_LANGUAGES = [
        "", "en", "zh", "fr", "de", "es", "ja", "ko", "pt", "ru", "it", "unknown",
    ]
    LEGO_TRACKS = [
        "", "vocals", "backing_vocals", "drums", "bass", "guitar",
        "keyboard", "percussion", "strings", "synth", "fx", "brass", "woodwinds",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models": (
                    "ACESTEP_MODELS",
                    {"tooltip": "Model paths from the AcestepCPP Model Loader node"},
                ),
                "caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Upbeat pop rock with driving guitars and catchy hooks",
                        "tooltip": (
                            "Natural language description of the music style, mood, "
                            "instruments, etc. Fed to both the LM and the DiT text encoder."
                        ),
                    },
                ),
            },
            "optional": {
                # ---- Lyrics / vocal content ----
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Song lyrics. Leave empty for the LM to generate them. "
                            "Set to '[Instrumental]' (or enable the instrumental toggle) "
                            "for no vocals."
                        ),
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Convenience toggle: when enabled and lyrics is empty, "
                            "sets lyrics to '[Instrumental]' so the DiT generates no vocals."
                        ),
                        "label_on": "Instrumental",
                        "label_off": "Vocal",
                    },
                ),
                # ---- Music metadata (LLM-filled when unset) ----
                "vocal_language": (
                    cls.VOCAL_LANGUAGES,
                    {
                        "default": "",
                        "tooltip": (
                            "BCP-47 language code for lyrics (e.g. 'en', 'fr', 'ja'). "
                            "Leave empty for the LM to detect. "
                            "Set to 'unknown' for an explicit no-language signal."
                        ),
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 1.0,
                        "tooltip": (
                            "Target audio duration in seconds. "
                            "0 lets the LM decide (clamped to [1, 600] s after generation)."
                        ),
                    },
                ),
                "bpm": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "tooltip": "Beats per minute (0 lets the LM decide)",
                    },
                ),
                "keyscale": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Musical key and scale, e.g. 'C major' or 'F# minor'. "
                            "Leave empty for the LM to decide."
                        ),
                    },
                ),
                "timesignature": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Time signature numerator, e.g. '4' for 4/4, '3' for 3/4. "
                            "Leave empty for the LM to decide."
                        ),
                    },
                ),
                # ---- DiT flow-matching ----
                "inference_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 200,
                        "tooltip": (
                            "Number of DiT denoising steps. "
                            "Turbo preset: 8. SFT preset: 50."
                        ),
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "CFG scale for the DiT. 0.0 is auto-resolved to 1.0 at runtime "
                            "(CFG disabled). Values > 1.0 on a turbo model are overridden to 1.0."
                        ),
                    },
                ),
                "shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "Flow-matching schedule shift. "
                            "Turbo preset: 3.0. SFT preset: 1.0."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "tooltip": "Random seed (-1 picks a random seed at runtime)",
                    },
                ),
                # ---- LM sampling ----
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": 0.85,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "LM sampling temperature for both phase 1 (lyrics/metadata) "
                            "and phase 2 (audio codes). Lower = more deterministic."
                        ),
                    },
                ),
                "lm_cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "LM classifier-free guidance scale. Active in phase 2 and in "
                            "phase 1 when lyrics are provided. 1.0 disables CFG."
                        ),
                    },
                ),
                "lm_top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "LM nucleus (top-p) sampling cutoff. 1.0 disables.",
                    },
                ),
                "lm_top_k": (
                    "STRING",
                    {
                        "default": "0",
                        "tooltip": (
                            "LM top-k sampling. 0 disables hard top-k (top_p still applies). "
                            "Accepts integers. Leave empty to use the default (0)."
                        ),
                    },
                ),
                "lm_negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Negative caption for LM CFG in phase 2. "
                            "Empty falls back to a caption-less unconditional prompt."
                        ),
                    },
                ),
                "use_cot_caption": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "When True, the LM enriches the caption via CoT and the enriched "
                            "version is fed to the DiT. When False, the user caption is used verbatim."
                        ),
                        "label_on": "Enabled",
                        "label_off": "Disabled",
                    },
                ),
                # ---- Source audio (cover / repaint / lego) ----
                "audio_cover_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Fraction of DiT steps that use the source audio as context "
                            "(only with src_audio). 0.0 = pure text-to-music, 1.0 = near-passthrough."
                        ),
                    },
                ),
                "repainting_start": (
                    "STRING",
                    {
                        "default": "-1.0",
                        "tooltip": (
                            "Repaint region start in seconds (requires src_audio). "
                            "-1 = inactive (0s when repaint_end is set). "
                            "Accepts floats. Leave empty to use the default (-1.0)."
                        ),
                    },
                ),
                "repainting_end": (
                    "STRING",
                    {
                        "default": "-1.0",
                        "tooltip": (
                            "Repaint region end in seconds (requires src_audio). "
                            "-1 = inactive (source duration when repaint_start is set). "
                            "Accepts floats. Leave empty to use the default (-1.0)."
                        ),
                    },
                ),
                "lego": (
                    cls.LEGO_TRACKS,
                    {
                        "default": "",
                        "tooltip": (
                            "Lego mode: generate one instrument track layered over an "
                            "existing backing track (requires src_audio and the base model). "
                            "Leave empty to disable."
                        ),
                    },
                ),
                # ---- Source audio path (string widget) ----
                "src_audio": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to a WAV or MP3 source file. Used for cover, repaint, "
                            "and lego modes. acestep.cpp reads the file natively — "
                            "no conversion needed."
                        ),
                    },
                ),
                # ---- LoRA adapter (convenience widgets) ----
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to a DiT LoRA adapter file (.safetensors or PEFT directory). "
                            "Use the LoRA Loader node instead for validated path input."
                        ),
                    },
                ),
                "lora_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "LoRA adapter scale",
                    },
                ),
                # ---- Node connections ----
                "lora": (
                    "ACESTEP_LORA",
                    {
                        "tooltip": (
                            "LoRA adapter from the Acestep.cpp LoRA Loader node. "
                            "Overrides 'lora_path' / 'lora_scale' when connected."
                        ),
                    },
                ),
                "options": (
                    "ACESTEP_OPTIONS",
                    {
                        "tooltip": (
                            "Advanced options from the Acestep.cpp Options node. "
                            "Controls output format, VAE tiling, batch size, and debug flags."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "AcestepCPP"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        lm_top_p=0.9,
        lm_top_k="0",
        audio_cover_strength=0.5,
        repainting_start="-1.0",
        repainting_end="-1.0",
        **kwargs,
    ):
        """Validate numeric optional inputs and reject non-parseable strings.

        ``lm_top_k``, ``repainting_start``, and ``repainting_end`` are declared
        as STRING in INPUT_TYPES so that ComfyUI's validation layer accepts the
        empty string ``""`` that older serialised workflows may store for those
        fields (``str("")`` succeeds where ``int("")`` / ``float("")`` would
        raise).  ``_coerce_int`` / ``_coerce_float`` inside ``generate()``
        convert ``""`` to the numeric default at runtime.

        ``lm_top_p`` and ``audio_cover_strength`` remain FLOAT widgets; they
        are listed here so that any non-empty string that cannot be parsed as
        a float is caught and reported before the node runs.

        Note: ComfyUI's VALIDATE_INPUTS bypasses *range* (min/max) checking
        for listed inputs; it does **not** bypass type conversion — that is
        why the three new fields use STRING rather than INT/FLOAT.
        """
        for name, val, coerce in (
            ("lm_top_p", lm_top_p, float),
            ("lm_top_k", lm_top_k, int),
            ("audio_cover_strength", audio_cover_strength, float),
            ("repainting_start", repainting_start, float),
            ("repainting_end", repainting_end, float),
        ):
            if isinstance(val, str) and val.strip():
                try:
                    coerce(val)
                except ValueError:
                    return f"Invalid value for {name}: {val!r}"
        return True

    def generate(
        self,
        models: Dict[str, Any],
        caption: str,
        lyrics: str = "",
        instrumental: bool = False,
        vocal_language: str = "",
        duration: float = 0.0,
        bpm: int = 0,
        keyscale: str = "",
        timesignature: str = "",
        inference_steps: int = 8,
        guidance_scale: float = 0.0,
        shift: float = 3.0,
        seed: int = -1,
        lm_temperature: float = 0.85,
        lm_cfg_scale: float = 2.0,
        lm_top_p: float = 0.9,
        lm_top_k: Union[str, int] = "0",
        lm_negative_prompt: str = "",
        use_cot_caption: bool = True,
        audio_cover_strength: float = 0.5,
        repainting_start: Union[str, float] = "-1.0",
        repainting_end: Union[str, float] = "-1.0",
        lego: str = "",
        src_audio: str = "",
        lora_path: str = "",
        lora_scale: float = 1.0,
        lora: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        # Merge options dict (from AcestepCPPOptions) with defaults.
        opts: Dict[str, Any] = options or {}

        # Coerce optional numeric inputs that may arrive as empty strings from
        # workflows saved with an older version of the node schema.
        lm_top_p = _coerce_float(lm_top_p, 0.9)
        lm_top_k = _coerce_int(lm_top_k, 0)
        audio_cover_strength = _coerce_float(audio_cover_strength, 0.5)
        repainting_start = _coerce_float(repainting_start, -1.0)
        repainting_end = _coerce_float(repainting_end, -1.0)

        # Apply instrumental convenience toggle: overrides lyrics to [Instrumental]
        # only when the lyrics field is empty (user-provided lyrics take priority).
        if instrumental and not lyrics.strip():
            lyrics = "[Instrumental]"

        ace_qwen3 = get_binary_path("ace-qwen3")
        dit_vae = get_binary_path("dit-vae")

        if not ace_qwen3:
            raise FileNotFoundError(
                "ace-qwen3 binary not found. "
                "Use the 'Acestep.cpp Builder' node to build it automatically, "
                "or run install.py from the node directory."
            )
        if not dit_vae:
            raise FileNotFoundError(
                "dit-vae binary not found. "
                "Use the 'Acestep.cpp Builder' node to build it automatically, "
                "or run install.py from the node directory."
            )

        # LoRA Loader node takes priority over freeform widgets.
        if lora is not None:
            lora_path = lora["path"]
            lora_scale = lora["scale"]

        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = os.path.join(tmpdir, "request.json")

            # Build the request JSON following the acestep.cpp reference:
            # https://github.com/ServeurpersoCom/acestep.cpp#request-json-reference
            request: Dict[str, Any] = {
                "caption": caption,
                "lyrics": lyrics,
                "bpm": bpm,
                "duration": duration,
                "vocal_language": vocal_language,
                "seed": seed,
                "lm_temperature": lm_temperature,
                "lm_cfg_scale": lm_cfg_scale,
                "lm_top_p": lm_top_p,
                "lm_top_k": lm_top_k,
                "lm_negative_prompt": lm_negative_prompt,
                "use_cot_caption": use_cot_caption,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "audio_cover_strength": audio_cover_strength,
            }

            # Optional metadata fields: only include when explicitly set so
            # the LM fills them via CoT when absent.
            if keyscale.strip():
                request["keyscale"] = keyscale.strip()
            if timesignature.strip():
                request["timesignature"] = timesignature.strip()

            # Repaint region (only meaningful with --src-audio)
            if repainting_start >= 0.0:
                request["repainting_start"] = repainting_start
            if repainting_end >= 0.0:
                request["repainting_end"] = repainting_end

            # Lego mode (requires --src-audio and base model)
            if lego.strip():
                request["lego"] = lego.strip()

            with open(request_path, "w") as f:
                json.dump(request, f)

            # ----------------------------------------------------------------
            # Step 1 – LM: ace-qwen3 → request0.json (request1.json …)
            # ----------------------------------------------------------------
            lm_batch: int = int(opts.get("lm_batch", 1))
            lm_cmd = [
                ace_qwen3,
                "--request", request_path,
                "--model", models["lm_model"],
            ]
            if lm_batch > 1:
                lm_cmd += ["--batch", str(lm_batch)]
            if opts.get("no_flash_attn", False):
                lm_cmd += ["--no-fa"]
            lm_max_seq: int = int(opts.get("lm_max_seq", 8192))
            if lm_max_seq != 8192:
                lm_cmd += ["--max-seq", str(lm_max_seq)]
            if opts.get("lm_no_fsm", False):
                lm_cmd += ["--no-fsm"]

            lm_result = subprocess.run(
                lm_cmd, capture_output=True, encoding='utf-8', errors='replace', cwd=tmpdir
            )
            if lm_result.returncode != 0:
                raise RuntimeError(
                    f"ace-qwen3 failed (exit {lm_result.returncode}):\n"
                    f"{lm_result.stderr}"
                )

            lm_output = os.path.join(tmpdir, "request0.json")
            if not os.path.isfile(lm_output):
                raise RuntimeError(
                    "ace-qwen3 did not produce request0.json.\n"
                    f"stdout: {lm_result.stdout}\nstderr: {lm_result.stderr}"
                )

            # ----------------------------------------------------------------
            # Step 2 – DiT+VAE: dit-vae → request00.{mp3|wav}
            # ----------------------------------------------------------------
            output_format: str = opts.get("output_format", "mp3")
            mp3_bitrate: int = int(opts.get("mp3_bitrate", 128))
            vae_chunk: int = int(opts.get("vae_chunk", 256))
            vae_overlap: int = int(opts.get("vae_overlap", 64))
            dit_batch: int = int(opts.get("dit_batch", 1))

            dit_cmd = [
                dit_vae,
                "--request", lm_output,
                "--text-encoder", models["text_encoder"],
                "--dit", models["dit"],
                "--vae", models["vae"],
                "--vae-chunk", str(vae_chunk),
                "--vae-overlap", str(vae_overlap),
            ]

            # Source audio (cover, repaint, lego modes)
            if src_audio.strip():
                dit_cmd += ["--src-audio", src_audio.strip()]

            # LoRA adapter
            if lora_path.strip():
                dit_cmd += ["--lora", lora_path.strip(), "--lora-scale", str(lora_scale)]

            # Output format
            if output_format == "wav":
                dit_cmd += ["--wav"]
            else:
                dit_cmd += ["--mp3-bitrate", str(mp3_bitrate)]

            # Batch
            if dit_batch > 1:
                dit_cmd += ["--batch", str(dit_batch)]

            # Flash attention
            if opts.get("no_flash_attn", False):
                dit_cmd += ["--no-fa"]

            dit_result = subprocess.run(
                dit_cmd, capture_output=True, encoding='utf-8', errors='replace', cwd=tmpdir
            )
            if dit_result.returncode != 0:
                raise RuntimeError(
                    f"dit-vae failed (exit {dit_result.returncode}):\n"
                    f"{dit_result.stderr}"
                )

            # Locate the first output file (request00.mp3 or request00.wav)
            ext = ".wav" if output_format == "wav" else ".mp3"
            audio_path = os.path.join(tmpdir, f"request00{ext}")
            if not os.path.isfile(audio_path):
                raise RuntimeError(
                    f"dit-vae did not produce request00{ext}.\n"
                    f"stdout: {dit_result.stdout}\nstderr: {dit_result.stderr}"
                )

            # Copy the native MP3/WAV to ComfyUI's output directory unchanged —
            # no decoding, no re-encoding, no external tools.
            out_dir = folder_paths.get_output_directory()
            # get_save_image_path is ComfyUI's generic counter-based filename
            # helper — it works for any file type despite the "image" name.
            full_out_folder, base_name, counter, subfolder, _ = (
                folder_paths.get_save_image_path("acestep", out_dir)
            )
            save_filename = f"{base_name}_{counter:05}{ext}"
            save_path = os.path.join(full_out_folder, save_filename)
            shutil.copy2(audio_path, save_path)

        # Return the absolute path to the generated audio file so downstream
        # nodes (e.g. AudioLoader) can load it directly by file path.
        return {
            "result": (save_path,),
            "ui": {"audio": [{"filename": save_filename, "subfolder": subfolder, "type": "output"}]},
        }
