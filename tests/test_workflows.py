"""Tests for workflow example JSON files.

Validates structure, model metadata, output nodes, and widget value types
without requiring ComfyUI to be installed.
"""

import glob
import json
import os

import pytest

WORKFLOW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workflow-examples",
)
WORKFLOW_FILES = sorted(glob.glob(os.path.join(WORKFLOW_DIR, "*.json")))

EXPECTED_MODEL_NAMES = {
    "acestep-5Hz-lm-4B-Q8_0.gguf",
    "Qwen3-Embedding-0.6B-Q8_0.gguf",
    "acestep-v15-turbo-Q8_0.gguf",
    "vae-BF16.gguf",
}

HF_BASE = "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _nodes_by_type(wf, node_type):
    return [n for n in wf["nodes"] if n["type"] == node_type]


# ---------------------------------------------------------------------------
# Parametrised: every workflow file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "workflow_path",
    WORKFLOW_FILES,
    ids=lambda p: os.path.basename(p),
)
class TestWorkflowStructure:

    # --- Basic structure ---------------------------------------------------

    def test_valid_json(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf, dict)

    def test_has_nodes_list(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf.get("nodes"), list)
        assert len(wf["nodes"]) >= 2

    def test_has_links_list(self, workflow_path):
        wf = _load(workflow_path)
        assert isinstance(wf.get("links"), list)

    def test_has_version(self, workflow_path):
        wf = _load(workflow_path)
        assert "version" in wf

    # --- Required custom nodes present ------------------------------------

    def test_has_model_loader_node(self, workflow_path):
        wf = _load(workflow_path)
        assert _nodes_by_type(wf, "AcestepCPPModelLoader"), \
            "Workflow must include AcestepCPPModelLoader"

    def test_has_generate_node(self, workflow_path):
        wf = _load(workflow_path)
        assert _nodes_by_type(wf, "AcestepCPPGenerate"), \
            "Workflow must include AcestepCPPGenerate"

    # --- Output: generate node is itself the output node ---------------------

    def test_has_no_audio_output_node(self, workflow_path):
        """AcestepCPPGenerate is itself an OUTPUT_NODE; no SaveAudio or PreviewAudio needed."""
        wf = _load(workflow_path)
        node_types = [n["type"] for n in wf["nodes"]]
        assert "SaveAudio" not in node_types, \
            "SaveAudio is no longer needed; AcestepCPPGenerate copies the file and shows its own player"
        assert "PreviewAudio" not in node_types, \
            "PreviewAudio is no longer needed; AcestepCPPGenerate shows its own audio player"

    def test_generate_has_no_output_links(self, workflow_path):
        """AcestepCPPGenerate is a terminal output node; its output slot must have no outgoing links."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            for output in node.get("outputs", []):
                assert not output.get("links"), \
                    f"AcestepCPPGenerate output '{output.get('name')}' should have no outgoing links"

    # --- Model loader widget values ---------------------------------------

    def test_model_loader_has_four_widget_values(self, workflow_path):
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            assert len(node.get("widgets_values", [])) == 4, \
                "AcestepCPPModelLoader needs exactly 4 widget values"

    def test_model_loader_widget_values_are_known_models(self, workflow_path):
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for v in node.get("widgets_values", []):
                assert v in EXPECTED_MODEL_NAMES, \
                    f"Unexpected model name in ModelLoader: {v!r}"

    # --- Generate node widget value types --------------------------------
    # Widget order (must match INPUT_TYPES — connection-type inputs are excluded):
    # 0:caption  1:lyrics  2:instrumental  3:vocal_language  4:duration  5:bpm
    # 6:keyscale  7:timesignature  8:inference_steps  9:guidance_scale  10:shift
    # 11:seed  12:lm_temperature  13:lm_cfg_scale  14:lm_top_p  15:lm_top_k
    # 16:lm_negative_prompt  17:use_cot_caption  18:audio_cover_strength
    # 19:repainting_start  20:repainting_end  21:lego  22:src_audio
    # 23:lora_path  24:lora_scale

    def test_generate_widget_count(self, workflow_path):
        """AcestepCPPGenerate must have exactly 25 widget values.

        Widget inputs (non-connection-type): caption, lyrics, instrumental,
        vocal_language, duration, bpm, keyscale, timesignature, inference_steps,
        guidance_scale, shift, seed, lm_temperature, lm_cfg_scale, lm_top_p,
        lm_top_k, lm_negative_prompt, use_cot_caption, audio_cover_strength,
        repainting_start, repainting_end, lego, src_audio, lora_path, lora_scale.
        """
        EXPECTED_WIDGET_COUNT = 25
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            assert len(wv) == EXPECTED_WIDGET_COUNT, \
                f"Expected {EXPECTED_WIDGET_COUNT} widget values, got {len(wv)}: {wv}"

    def test_generate_lm_top_p_is_numeric(self, workflow_path):
        """Widget index 14 (lm_top_p) must be a number, never an empty string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 14:
                assert isinstance(wv[14], (int, float)), \
                    f"lm_top_p (index 14) should be numeric, got {type(wv[14])}"

    def test_generate_lm_top_k_is_valid(self, workflow_path):
        """Widget index 15 (lm_top_k) must be a non-empty parseable integer string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 15:
                val = wv[15]
                assert val != "", \
                    f"lm_top_k (index 15) must not be an empty string, got {val!r}"
                try:
                    int(str(val))
                except (ValueError, TypeError):
                    raise AssertionError(
                        f"lm_top_k (index 15) must be parseable as int, got {val!r}"
                    )

    def test_generate_audio_cover_strength_is_numeric(self, workflow_path):
        """Widget index 18 (audio_cover_strength) must be a number."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 18:
                assert isinstance(wv[18], (int, float)), \
                    f"audio_cover_strength (index 18) should be numeric, got {type(wv[18])}"

    def test_generate_repainting_start_is_valid(self, workflow_path):
        """Widget index 19 (repainting_start) must be a non-empty parseable float string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 19:
                val = wv[19]
                assert val != "", \
                    f"repainting_start (index 19) must not be an empty string, got {val!r}"
                try:
                    float(str(val))
                except (ValueError, TypeError):
                    raise AssertionError(
                        f"repainting_start (index 19) must be parseable as float, got {val!r}"
                    )

    def test_generate_repainting_end_is_valid(self, workflow_path):
        """Widget index 20 (repainting_end) must be a non-empty parseable float string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 20:
                val = wv[20]
                assert val != "", \
                    f"repainting_end (index 20) must not be an empty string, got {val!r}"
                try:
                    float(str(val))
                except (ValueError, TypeError):
                    raise AssertionError(
                        f"repainting_end (index 20) must be parseable as float, got {val!r}"
                    )

    def test_generate_src_audio_is_string(self, workflow_path):
        """Widget index 22 (src_audio) must be a string."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            wv = node.get("widgets_values", [])
            if len(wv) > 22:
                assert isinstance(wv[22], str), \
                    f"src_audio (index 22) should be a string, got {type(wv[22])}"

    def test_generate_no_task_type(self, workflow_path):
        """task_type is not a valid acestep.cpp field — must not appear in nodes."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPGenerate"):
            # widget index 2 should be instrumental (bool), not a task_type string
            wv = node.get("widgets_values", [])
            if len(wv) > 2:
                assert not (isinstance(wv[2], str) and wv[2] in {"text2music", "cover", "repaint"}), \
                    "task_type string found at widget index 2 — node should use new API"

    # --- extra.models download metadata ----------------------------------

    def test_extra_models_present(self, workflow_path):
        wf = _load(workflow_path)
        models = wf.get("extra", {}).get("models", [])
        assert len(models) == 4, \
            f"Expected 4 entries in extra.models, got {len(models)}"

    def test_extra_models_names_match_expected(self, workflow_path):
        wf = _load(workflow_path)
        names = {m["name"] for m in wf.get("extra", {}).get("models", [])}
        assert names == EXPECTED_MODEL_NAMES

    def test_extra_models_have_huggingface_urls(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            assert m.get("url", "").startswith(HF_BASE), \
                f"{m.get('name')} has unexpected URL: {m.get('url')}"

    def test_extra_models_save_path_in_text_encoders(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            assert m.get("save_path", "").startswith("text_encoders/"), \
                f"{m.get('name')} save_path should be under text_encoders/"

    def test_extra_models_save_path_matches_name(self, workflow_path):
        wf = _load(workflow_path)
        for m in wf.get("extra", {}).get("models", []):
            expected = f"text_encoders/{m['name']}"
            assert m.get("save_path") == expected, \
                f"save_path mismatch for {m['name']}: {m.get('save_path')}"

    # --- Node properties models (ComfyUI auto-download) ------------------

    def test_model_loader_properties_has_models(self, workflow_path):
        """AcestepCPPModelLoader.properties must include a 'models' list."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            models = node.get("properties", {}).get("models")
            assert isinstance(models, list) and len(models) == 4, \
                "AcestepCPPModelLoader properties must have exactly 4 model entries"

    def test_model_loader_properties_models_names(self, workflow_path):
        """Models in node properties must match expected GGUF filenames."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            names = {m["name"] for m in node["properties"].get("models", [])}
            assert names == EXPECTED_MODEL_NAMES

    def test_model_loader_properties_models_urls(self, workflow_path):
        """Models in node properties must have HuggingFace download URLs."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("url", "").startswith(HF_BASE), \
                    f"{m.get('name')} has unexpected URL: {m.get('url')}"

    def test_model_loader_properties_models_directory(self, workflow_path):
        """Models in node properties must specify 'text_encoders' directory."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("directory") == "text_encoders", \
                    f"{m.get('name')} directory should be 'text_encoders', got {m.get('directory')!r}"

    def test_model_loader_properties_models_url_matches_name(self, workflow_path):
        """Each model's URL must end with its filename."""
        wf = _load(workflow_path)
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            for m in node["properties"].get("models", []):
                assert m.get("url", "").endswith(m["name"]), \
                    f"URL does not end with filename for {m['name']}: {m.get('url')}"

    # --- Graph connectivity -----------------------------------------------

    def test_model_loader_output_connected(self, workflow_path):
        """The ACESTEP_MODELS output of ModelLoader must be connected."""
        wf = _load(workflow_path)
        link_src_nodes = {lnk[1] for lnk in wf.get("links", [])}
        for node in _nodes_by_type(wf, "AcestepCPPModelLoader"):
            assert node["id"] in link_src_nodes, \
                "AcestepCPPModelLoader output is not connected to any node"
