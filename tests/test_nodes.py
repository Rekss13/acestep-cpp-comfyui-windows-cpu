"""Unit tests for nodes.py — covers node helpers, INPUT_TYPES, and validation
logic.  All ComfyUI / torch imports are stubbed via tests/conftest.py so the
suite runs in any plain Python environment."""

import os
import sys
import tempfile

import pytest

import nodes  # pre-loaded into sys.modules by tests/conftest.py


# ===========================================================================
# _coerce_float
# ===========================================================================

class TestCoerceFloat:
    def test_float_passthrough(self):
        assert nodes._coerce_float(0.9, 0.5) == pytest.approx(0.9)

    def test_int_converts(self):
        assert nodes._coerce_float(1, 0.0) == pytest.approx(1.0)

    def test_valid_string_converts(self):
        assert nodes._coerce_float("0.75", 0.0) == pytest.approx(0.75)

    def test_empty_string_uses_default(self):
        assert nodes._coerce_float("", 0.9) == pytest.approx(0.9)

    def test_whitespace_string_uses_default(self):
        assert nodes._coerce_float("   ", 1.0) == pytest.approx(1.0)

    def test_different_defaults(self):
        assert nodes._coerce_float("", 0.0) == pytest.approx(0.0)
        assert nodes._coerce_float("", 2.5) == pytest.approx(2.5)


# ===========================================================================
# _coerce_int
# ===========================================================================

class TestCoerceInt:
    def test_int_passthrough(self):
        assert nodes._coerce_int(5, 0) == 5

    def test_float_truncates(self):
        assert nodes._coerce_int(3.9, 0) == 3

    def test_valid_string_converts(self):
        assert nodes._coerce_int("42", 0) == 42

    def test_empty_string_uses_default(self):
        assert nodes._coerce_int("", 7) == 7

    def test_whitespace_string_uses_default(self):
        assert nodes._coerce_int("   ", 10) == 10

    def test_different_defaults(self):
        assert nodes._coerce_int("", 0) == 0
        assert nodes._coerce_int("", -1) == -1


# ===========================================================================
# scan_gguf_models / get_merged_model_folders / find_model_path
# ===========================================================================

class TestScanGgufModels:
    def test_returns_list(self):
        result = nodes.scan_gguf_models()
        assert isinstance(result, list)

    def test_finds_gguf_files_via_folder_paths(self, tmp_path, monkeypatch):
        (tmp_path / "model-a.gguf").write_text("mock")
        (tmp_path / "model-b.gguf").write_text("mock")
        (tmp_path / "readme.txt").write_text("not a model")

        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert "model-a.gguf" in result
        assert "model-b.gguf" in result
        assert "readme.txt" not in result

    def test_result_is_sorted(self, tmp_path, monkeypatch):
        for name in ("z-model.gguf", "a-model.gguf", "m-model.gguf"):
            (tmp_path / name).write_text("mock")

        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert result == sorted(result)

    def test_falls_back_to_manual_scan(self, tmp_path, monkeypatch):
        """When folder_paths returns no files the manual scan path is used."""
        (tmp_path / "fallback.gguf").write_text("mock")

        # Ensure the registered list is empty (no acestep_gguf entry)
        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [])

        # Point manual scan at tmp_path via get_merged_model_folders
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        result = nodes.scan_gguf_models()
        assert "fallback.gguf" in result

    def test_no_models_returns_empty_list(self, monkeypatch):
        import folder_paths as fp
        monkeypatch.setitem(fp._registered, "acestep_gguf", [])
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [])

        result = nodes.scan_gguf_models()
        assert result == []


class TestFindModelPath:
    def test_finds_existing_model(self, tmp_path, monkeypatch):
        model = tmp_path / "test-model.gguf"
        model.write_text("mock")
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        result = nodes.find_model_path("test-model.gguf")
        assert result == str(model)

    def test_returns_none_for_missing_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nodes, "get_merged_model_folders", lambda: [str(tmp_path)])

        assert nodes.find_model_path("nonexistent.gguf") is None

    def test_searches_multiple_folders(self, tmp_path, monkeypatch):
        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir()
        folder_b.mkdir()
        (folder_b / "model.gguf").write_text("mock")

        monkeypatch.setattr(
            nodes, "get_merged_model_folders",
            lambda: [str(folder_a), str(folder_b)],
        )

        result = nodes.find_model_path("model.gguf")
        assert result == str(folder_b / "model.gguf")


# ===========================================================================
# AcestepCPPModelLoader
# ===========================================================================

class TestAcestepCPPModelLoaderInputTypes:
    def test_has_required_section(self):
        result = nodes.AcestepCPPModelLoader.INPUT_TYPES()
        assert "required" in result

    def test_has_four_model_fields(self):
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert set(req) == {"lm_model", "text_encoder_model", "dit_model", "vae_model"}

    def test_placeholder_when_no_models(self, monkeypatch):
        monkeypatch.setattr(nodes, "scan_gguf_models", lambda: [])
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert req["lm_model"][0] == ["No GGUF models found"]
        assert req["dit_model"][0] == ["No GGUF models found"]

    def test_model_names_appear_in_dropdown(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "scan_gguf_models",
            lambda: ["acestep-v15-turbo-Q8_0.gguf", "vae-BF16.gguf"],
        )
        req = nodes.AcestepCPPModelLoader.INPUT_TYPES()["required"]
        assert "acestep-v15-turbo-Q8_0.gguf" in req["dit_model"][0]
        assert "vae-BF16.gguf" in req["vae_model"][0]

    def test_return_types(self):
        assert nodes.AcestepCPPModelLoader.RETURN_TYPES == ("ACESTEP_MODELS",)

    def test_category(self):
        assert nodes.AcestepCPPModelLoader.CATEGORY == "AcestepCPP"


# ===========================================================================
# AcestepCPPLoraLoader
# ===========================================================================

class TestAcestepCPPLoraLoader:
    @pytest.fixture
    def loader(self):
        return nodes.AcestepCPPLoraLoader()

    def test_empty_path_raises_value_error(self, loader):
        with pytest.raises(ValueError, match="lora_path is empty"):
            loader.load_lora("", 1.0)

    def test_whitespace_path_raises_value_error(self, loader):
        with pytest.raises(ValueError, match="lora_path is empty"):
            loader.load_lora("   ", 1.0)

    def test_unsupported_extension_raises(self, loader, tmp_path):
        f = tmp_path / "lora.bin"
        f.write_text("mock")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_lora(str(f), 1.0)

    def test_missing_file_raises_file_not_found(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load_lora(str(tmp_path / "missing.gguf"), 1.0)

    def test_valid_gguf_returns_correct_dict(self, loader, tmp_path):
        f = tmp_path / "lora.gguf"
        f.write_text("mock")
        result = loader.load_lora(str(f), 0.8)
        assert result == ({"path": str(f), "scale": 0.8},)

    def test_valid_safetensors_accepted(self, loader, tmp_path):
        f = tmp_path / "lora.safetensors"
        f.write_text("mock")
        result = loader.load_lora(str(f), 1.0)
        assert result[0]["path"] == str(f)

    def test_return_types(self):
        assert nodes.AcestepCPPLoraLoader.RETURN_TYPES == ("ACESTEP_LORA",)


# ===========================================================================
# AcestepCPPGenerate — INPUT_TYPES and coercion
# ===========================================================================

class TestAcestepCPPGenerateInputTypes:
    def test_required_has_models_and_caption(self):
        req = nodes.AcestepCPPGenerate.INPUT_TYPES()["required"]
        assert "models" in req
        assert "caption" in req

    def test_optional_contains_float_fields(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "lm_top_p" in opt
        assert "audio_cover_strength" in opt

    def test_lm_top_p_default_in_range(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        spec = opt["lm_top_p"][1]
        assert 0.0 <= spec["default"] <= 1.0

    def test_audio_cover_strength_default_in_range(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        spec = opt["audio_cover_strength"][1]
        assert 0.0 <= spec["default"] <= 1.0

    def test_optional_connections_present(self):
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        # src_audio_input removed; binary supports WAV/MP3 natively via src_audio path
        assert "lora" in opt
        assert "options" in opt

    def test_src_audio_after_lego_in_widget_order(self):
        """src_audio must appear after lego in INPUT_TYPES so the widget
        positional index matches the saved workflow files (index 22)."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        # Connection-type inputs do not produce widget slots in workflows.
        _CONNECTION_TYPES = {"ACESTEP_MODELS", "ACESTEP_LORA", "ACESTEP_OPTIONS"}
        widget_names = []
        for name, spec in opt.items():
            type_val = spec[0] if isinstance(spec, tuple) else spec
            # list/tuple values mean a dropdown — always a widget input
            if isinstance(type_val, str) and type_val in _CONNECTION_TYPES:
                continue
            widget_names.append(name)
        lego_idx = widget_names.index("lego")
        src_audio_idx = widget_names.index("src_audio")
        assert src_audio_idx > lego_idx, (
            f"src_audio (index {src_audio_idx}) must come after lego "
            f"(index {lego_idx}) in INPUT_TYPES optional section"
        )

    def test_optional_has_new_params(self):
        """New params added in the redesign must all be present."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "lm_top_k" in opt
        assert "use_cot_caption" in opt
        assert "repainting_start" in opt
        assert "repainting_end" in opt
        assert "lego" in opt

    def test_lm_top_k_is_string_type(self):
        """lm_top_k must be STRING so ComfyUI never runs int('') on stale workflows."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert opt["lm_top_k"][0] == "STRING", (
            "lm_top_k must be STRING (not INT) so that empty-string widget values "
            "from older workflows do not trigger ComfyUI's int('') coercion error"
        )

    def test_repainting_start_is_string_type(self):
        """repainting_start must be STRING so ComfyUI never runs float('') on stale workflows."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert opt["repainting_start"][0] == "STRING", (
            "repainting_start must be STRING (not FLOAT) so that empty-string widget "
            "values from older workflows do not trigger ComfyUI's float('') coercion error"
        )

    def test_repainting_end_is_string_type(self):
        """repainting_end must be STRING so ComfyUI never runs float('') on stale workflows."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert opt["repainting_end"][0] == "STRING", (
            "repainting_end must be STRING (not FLOAT) so that empty-string widget "
            "values from older workflows do not trigger ComfyUI's float('') coercion error"
        )

    def test_guidance_scale_default_is_zero(self):
        """guidance_scale default must be 0.0 (auto-resolved by the binary)."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert opt["guidance_scale"][1]["default"] == 0.0

    def test_duration_is_float(self):
        """duration must be a FLOAT with 0.0 meaning 'unset'."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert opt["duration"][0] == "FLOAT"
        assert opt["duration"][1]["default"] == 0.0

    def test_task_type_removed(self):
        """task_type is not a valid acestep.cpp JSON field and must be absent."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "task_type" not in opt

    def test_reference_audio_removed(self):
        """reference_audio / reference_audio_input consolidated into src_audio."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "reference_audio" not in opt
        assert "reference_audio_input" not in opt

    def test_return_types(self):
        assert nodes.AcestepCPPGenerate.RETURN_TYPES == ("STRING",)

    def test_return_names(self):
        assert nodes.AcestepCPPGenerate.RETURN_NAMES == ("filepath",)

    def test_is_output_node(self):
        assert nodes.AcestepCPPGenerate.OUTPUT_NODE is True

    def test_no_audio_tensor_input(self):
        """src_audio_input AUDIO connection was removed — binary reads WAV/MP3
        natively so no Python tensor conversion is needed on the input side."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "src_audio_input" not in opt

    def test_audio_tensor_output(self):
        """Generate must expose a STRING (filepath) output so AudioLoader nodes
        can receive the generated audio file path."""
        assert nodes.AcestepCPPGenerate.RETURN_TYPES == ("STRING",)
        assert nodes.AcestepCPPGenerate.OUTPUT_NODE is True

    def test_lego_tracks_list(self):
        """LEGO_TRACKS must include the track names documented in the README."""
        assert "guitar" in nodes.AcestepCPPGenerate.LEGO_TRACKS
        assert "drums" in nodes.AcestepCPPGenerate.LEGO_TRACKS
        assert "" in nodes.AcestepCPPGenerate.LEGO_TRACKS


# ===========================================================================
# AcestepCPPGenerate — VALIDATE_INPUTS
# ===========================================================================

class TestValidateInputs:
    """VALIDATE_INPUTS must accept empty strings and reject non-parseable non-empty
    strings for the numeric optional fields.

    ``lm_top_k``, ``repainting_start``, and ``repainting_end`` are now STRING
    widgets (so ComfyUI's own ``int()``/``float()`` coercion never fires on
    them) while VALIDATE_INPUTS still guards against non-parseable non-empty
    string values.  ``lm_top_p`` and ``audio_cover_strength`` remain FLOAT;
    they are in the signature so that stale empty-string values from very old
    workflows can be caught with a helpful error rather than an exception."""

    def _vi(self, **kwargs):
        return nodes.AcestepCPPGenerate.VALIDATE_INPUTS(**kwargs)

    # ---- lm_top_p (FLOAT) ------------------------------------------------
    def test_lm_top_p_empty_string_accepted(self):
        assert self._vi(lm_top_p="") is True

    def test_lm_top_p_valid_string_accepted(self):
        assert self._vi(lm_top_p="0.9") is True

    def test_lm_top_p_invalid_string_rejected(self):
        result = self._vi(lm_top_p="abc")
        assert isinstance(result, str) and "lm_top_p" in result

    # ---- lm_top_k (INT) --------------------------------------------------
    def test_lm_top_k_empty_string_accepted(self):
        assert self._vi(lm_top_k="") is True

    def test_lm_top_k_numeric_accepted(self):
        assert self._vi(lm_top_k=0) is True

    def test_lm_top_k_valid_string_accepted(self):
        assert self._vi(lm_top_k="50") is True

    def test_lm_top_k_invalid_string_rejected(self):
        result = self._vi(lm_top_k="bad")
        assert isinstance(result, str) and "lm_top_k" in result

    # ---- audio_cover_strength (FLOAT) ------------------------------------
    def test_audio_cover_strength_empty_string_accepted(self):
        assert self._vi(audio_cover_strength="") is True

    # ---- repainting_start (FLOAT) ----------------------------------------
    def test_repainting_start_empty_string_accepted(self):
        assert self._vi(repainting_start="") is True

    def test_repainting_start_numeric_accepted(self):
        assert self._vi(repainting_start=-1.0) is True

    def test_repainting_start_valid_string_accepted(self):
        assert self._vi(repainting_start="10.5") is True

    def test_repainting_start_invalid_string_rejected(self):
        result = self._vi(repainting_start="bad")
        assert isinstance(result, str) and "repainting_start" in result

    # ---- repainting_end (FLOAT) ------------------------------------------
    def test_repainting_end_empty_string_accepted(self):
        assert self._vi(repainting_end="") is True

    def test_repainting_end_numeric_accepted(self):
        assert self._vi(repainting_end=-1.0) is True

    def test_repainting_end_valid_string_accepted(self):
        assert self._vi(repainting_end="30.0") is True

    def test_repainting_end_invalid_string_rejected(self):
        result = self._vi(repainting_end="bad")
        assert isinstance(result, str) and "repainting_end" in result

    # ---- combined (all five at once) -------------------------------------
    def test_all_empty_strings_accepted(self):
        assert self._vi(
            lm_top_p="", lm_top_k="",
            audio_cover_strength="",
            repainting_start="", repainting_end="",
        ) is True

    def test_all_numeric_accepted(self):
        assert self._vi(
            lm_top_p=0.9, lm_top_k=0,
            audio_cover_strength=0.5,
            repainting_start=-1.0, repainting_end=-1.0,
        ) is True


# ===========================================================================
# AcestepCPPOptions
# ===========================================================================

class TestAcestepCPPOptions:
    @pytest.fixture
    def node(self):
        return nodes.AcestepCPPOptions()

    def test_return_types(self):
        assert nodes.AcestepCPPOptions.RETURN_TYPES == ("ACESTEP_OPTIONS",)

    def test_return_names(self):
        assert nodes.AcestepCPPOptions.RETURN_NAMES == ("options",)

    def test_category(self):
        assert nodes.AcestepCPPOptions.CATEGORY == "AcestepCPP"

    def test_has_output_format(self):
        opt = nodes.AcestepCPPOptions.INPUT_TYPES()["optional"]
        assert "output_format" in opt

    def test_has_vae_tiling_params(self):
        opt = nodes.AcestepCPPOptions.INPUT_TYPES()["optional"]
        assert "vae_chunk" in opt
        assert "vae_overlap" in opt

    def test_has_batch_params(self):
        opt = nodes.AcestepCPPOptions.INPUT_TYPES()["optional"]
        assert "lm_batch" in opt
        assert "dit_batch" in opt

    def test_has_debug_flags(self):
        opt = nodes.AcestepCPPOptions.INPUT_TYPES()["optional"]
        assert "no_flash_attn" in opt
        assert "lm_max_seq" in opt
        assert "lm_no_fsm" in opt

    def test_output_formats_include_mp3_and_wav(self):
        assert "mp3" in nodes.AcestepCPPOptions.OUTPUT_FORMATS
        assert "wav" in nodes.AcestepCPPOptions.OUTPUT_FORMATS

    def test_get_options_returns_dict(self, node):
        result = node.get_options(output_format="wav", mp3_bitrate=192)
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0]["output_format"] == "wav"
        assert result[0]["mp3_bitrate"] == 192

    def test_get_options_empty_returns_empty_dict(self, node):
        result = node.get_options()
        assert result == ({},)


# ===========================================================================
# AcestepCPPModelDownloader
# ===========================================================================

class TestAcestepCPPModelDownloader:
    def test_input_types_has_required_fields(self):
        req = nodes.AcestepCPPModelDownloader.INPUT_TYPES()["required"]
        assert "save_dir" in req
        assert "lm_size" in req
        assert "quant" in req
        assert "dit_variant" in req

    def test_is_output_node(self):
        assert nodes.AcestepCPPModelDownloader.OUTPUT_NODE is True


# ===========================================================================
# AcestepCPPBuilder
# ===========================================================================

class TestAcestepCPPBuilder:
    def test_is_output_node(self):
        assert nodes.AcestepCPPBuilder.OUTPUT_NODE is True

    def test_backends_list(self):
        assert "auto" in nodes.AcestepCPPBuilder.BACKENDS
        assert "cpu" in nodes.AcestepCPPBuilder.BACKENDS

    def test_detect_backend_returns_string(self):
        backend = nodes.AcestepCPPBuilder._detect_backend()
        assert isinstance(backend, str)
        assert backend in nodes.AcestepCPPBuilder.BACKENDS

    def test_cmake_flags_cpu(self):
        assert nodes.AcestepCPPBuilder._cmake_flags("cpu") == []

    def test_cmake_flags_cuda(self):
        assert "-DGGML_CUDA=ON" in nodes.AcestepCPPBuilder._cmake_flags("cuda")

    def test_cmake_flags_blas(self):
        assert "-DGGML_BLAS=ON" in nodes.AcestepCPPBuilder._cmake_flags("blas")


# ===========================================================================
# _binary_in_build — shared helper for multi-location binary search
# ===========================================================================

class TestBinaryInBuild:
    """_binary_in_build checks both build/ and build/bin/ (ggml default)."""

    def test_found_directly(self, tmp_path):
        binary = tmp_path / "ace-qwen3"
        binary.write_text("mock")
        assert nodes._binary_in_build(str(tmp_path), "ace-qwen3") == str(binary)

    def test_found_in_bin_subdir(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "ace-qwen3"
        binary.write_text("mock")
        assert nodes._binary_in_build(str(tmp_path), "ace-qwen3") == str(binary)

    def test_prefers_direct_over_bin(self, tmp_path):
        direct = tmp_path / "ace-qwen3"
        direct.write_text("direct")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "ace-qwen3").write_text("in_bin")
        assert nodes._binary_in_build(str(tmp_path), "ace-qwen3") == str(direct)

    def test_returns_none_when_absent(self, tmp_path):
        assert nodes._binary_in_build(str(tmp_path), "ace-qwen3") is None


# ===========================================================================
# get_binary_path — multi-location search
# ===========================================================================

class TestGetBinaryPath:
    """get_binary_path must honour explicit config paths and system PATH;
    the local build/ vs build/bin/ logic is delegated to _binary_in_build
    and covered by TestBinaryInBuild above."""

    def test_explicit_config_path_returned(self, tmp_path, monkeypatch):
        """Binary path from config.json binary_paths is returned directly."""
        binary = tmp_path / "ace-qwen3"
        binary.write_text("mock")
        monkeypatch.setattr(
            nodes, "load_config",
            lambda: {"binary_paths": {"ace-qwen3": str(binary)}},
        )
        assert nodes.get_binary_path("ace-qwen3") == str(binary)

    def test_explicit_config_path_missing_file_ignored(self, tmp_path, monkeypatch):
        """Config binary_paths entry is skipped when the file does not exist."""
        monkeypatch.setattr(
            nodes, "load_config",
            lambda: {"binary_paths": {"ace-qwen3": str(tmp_path / "nonexistent")}},
        )
        monkeypatch.setattr(nodes.shutil, "which", lambda *a, **kw: None)
        # No local build files either — result should be None
        result = nodes.get_binary_path("ace-qwen3")
        assert result is None

    def test_system_path_lookup(self, monkeypatch):
        """Binary found on the system PATH is returned."""
        monkeypatch.setattr(nodes, "load_config", lambda: {})
        monkeypatch.setattr(
            nodes.shutil, "which", lambda name, **kw: f"/usr/bin/{name}"
        )
        result = nodes.get_binary_path("ace-qwen3")
        assert result == "/usr/bin/ace-qwen3"

    def test_returns_none_when_nowhere(self, monkeypatch):
        """Returns None when binary is absent from config, PATH, and local build."""
        monkeypatch.setattr(nodes, "load_config", lambda: {})
        monkeypatch.setattr(nodes.shutil, "which", lambda *a, **kw: None)
        # Redirect the node __file__ into an empty temp dir so no local build
        # file is found.
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                nodes, "__file__", os.path.join(tmp, "nodes.py")
            )
            result = nodes.get_binary_path("ace-qwen3")
        assert result is None


# ===========================================================================
# No Python audio-decoding imports
# ===========================================================================

class TestNoPythonAudioProcessing:
    """These tests enforce that the node does not import Python audio-decoding
    libraries at module level (which would bloat startup time).  The binary
    handles WAV and MP3 natively on the input side (src_audio path) and writes
    output natively; the node simply returns the output file path as a STRING."""

    _AUDIO_LIBS = ["torchaudio", "torch", "wave", "numpy", "soundfile", "pydub"]

    def test_no_audio_decode_imports_at_module_level(self):
        """None of the Python audio-decoding libraries must appear as true
        module-level (non-indented) imports in nodes.py — they would be pulled
        in unconditionally at ComfyUI startup time.  Lazy imports *inside*
        functions are permitted."""
        with open(nodes.__file__) as f:
            nodes_src = f.read()
        # Only consider non-indented lines that are import statements
        # (indented imports are lazy/conditional and do not affect startup).
        module_level_import_lines = [
            ln
            for ln in nodes_src.splitlines()
            if (ln.startswith("import ") or ln.startswith("from "))
        ]
        for lib in self._AUDIO_LIBS:
            for ln in module_level_import_lines:
                assert lib not in ln, (
                    f"nodes.py imports audio-decoding library '{lib}' at module level: {ln!r}"
                )

    def test_src_audio_input_absent(self):
        """src_audio_input AUDIO connection must not exist — binary reads
        WAV/MP3 natively so no LoadAudio → tensor → torchaudio.save pipeline
        is needed."""
        opt = nodes.AcestepCPPGenerate.INPUT_TYPES()["optional"]
        assert "src_audio_input" not in opt

    def test_generate_has_filepath_output(self):
        """RETURN_TYPES must be STRING (filepath) so that AudioLoader and other
        file-path-aware nodes can be connected to the Generate node."""
        assert nodes.AcestepCPPGenerate.RETURN_TYPES == ("STRING",)


# ===========================================================================
# requirements.txt completeness
# ===========================================================================

class TestRequirementsTxt:
    """Every third-party package imported by nodes.py must be listed in
    requirements.txt so that ComfyUI Manager (and manual installs) know to
    pull it in."""

    # Packages that ComfyUI itself provides or that are part of the Python
    # standard library — they must not appear in requirements.txt.
    _COMFY_PROVIDED = {"folder_paths"}

    def _req_packages(self):
        req_path = os.path.join(os.path.dirname(nodes.__file__), "requirements.txt")
        with open(req_path) as f:
            return {
                line.strip().split("==")[0].split(">=")[0].split("~=")[0].lower()
                for line in f
                if line.strip() and not line.startswith("#")
            }

    def _nodes_third_party_imports(self):
        import ast
        import sys

        stdlib = set(sys.stdlib_module_names)
        stdlib |= self._COMFY_PROVIDED

        with open(nodes.__file__) as f:
            src = f.read()

        tree = ast.parse(src)
        pkgs = set()
        for node_ast in ast.walk(tree):
            if isinstance(node_ast, ast.Import):
                for alias in node_ast.names:
                    pkg = alias.name.split(".")[0]
                    if pkg not in stdlib:
                        pkgs.add(pkg.lower())
            elif isinstance(node_ast, ast.ImportFrom):
                pkg = (node_ast.module or "").split(".")[0]
                if pkg and pkg not in stdlib:
                    pkgs.add(pkg.lower())
        return pkgs

    def test_requirements_txt_exists(self):
        req_path = os.path.join(os.path.dirname(nodes.__file__), "requirements.txt")
        assert os.path.isfile(req_path), "requirements.txt must exist in the node directory"

    def test_all_third_party_imports_declared(self):
        """Every third-party package imported (lazily or otherwise) in nodes.py
        must appear in requirements.txt."""
        req = self._req_packages()
        missing = self._nodes_third_party_imports() - req
        assert not missing, (
            f"nodes.py imports packages not listed in requirements.txt: {sorted(missing)}"
        )
