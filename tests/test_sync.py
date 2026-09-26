"""Tests for dsh-copilot-sync.

Covers the 13 required scenarios:
1. same-named endpoint adds new models
2. existing models keep local config, not overwritten by source
3. default deletes target-only models and their settings entries
4. --no-delete only adds, never deletes
5. --dry-run writes nothing
6. source read failure leaves target completely unchanged
7. empty source model list skips deletion
8. non-customendpoint entries in target stay unchanged
9. vendor == "copilot" entries stay unchanged
10. duplicate source endpoint names raise an error
11. running twice reports no change the second time
12. atomic write skips when content is identical
13. invalid / non-array target JSON returns a config error

No test touches the network; every source and target is built in tmp_path.
"""

from __future__ import annotations

import json

import pytest

from dsh_copilot_sync.config import load_config, read_original_text, serialize, write_config_atomic
from dsh_copilot_sync.dsh_source import DshSourceIndex, load_dsh_source
from dsh_copilot_sync.models import DshEndpoint, DshModel, ModelSyncError, SyncOutcome
from dsh_copilot_sync.sync import sync_config


def _make_source(endpoint_name: str, models: list[dict], fallback_url: str | None = None) -> DshSourceIndex:
    """Build a DshSourceIndex directly, bypassing file parsing."""
    endpoint = DshEndpoint(
        name=endpoint_name,
        models=[DshModel.from_raw(m) for m in models],
        fallback_url=fallback_url,
    )
    return DshSourceIndex(endpoints={endpoint_name: endpoint})


def _custom_endpoint(
    name: str,
    models: list[dict] | None = None,
    settings: dict | None = None,
    **extra,
) -> dict:
    provider: dict = {
        "name": name,
        "vendor": "customendpoint",
        "apiKey": "${input:chat.lm.secret.abc}",
        "apiType": "chat-completions",
    }
    if models is not None:
        provider["models"] = models
    if settings is not None:
        provider["settings"] = settings
    provider.update(extra)
    return provider


def _model(id: str, **fields) -> dict:
    entry: dict = {"id": id}
    entry.update(fields)
    return entry


def _model_obj(
    id: str,
    *,
    url: str = "https://api.example.com",
    **fields,
) -> dict:
    entry: dict = {"id": id, "url": url, "toolCalling": True, "vision": True,
                   "maxInputTokens": 1000000, "maxOutputTokens": 384000,
                   "supportsReasoningEffort": ["max"]}
    entry.update(fields)
    return entry


# ---------------------------------------------------------------------------
# 1. same-named endpoint adds new models
# ---------------------------------------------------------------------------


def test_adds_new_models_to_matching_endpoint():
    source = _make_source("EndpointA", [
        {"id": "model-a", "name": "Model A", "url": "https://api.example.com"},
        {"id": "model-b"},
    ])
    target = [
        _custom_endpoint("EndpointA", models=[_model_obj("model-a", maxOutputTokens=999)]),
    ]
    outcome = sync_config(target, source, allow_delete=True)

    endpoint_a = outcome.config[0]
    ids = [m["id"] for m in endpoint_a["models"]]
    assert ids == ["model-a", "model-b"]
    kept_a = next(m for m in endpoint_a["models"] if m["id"] == "model-a")
    assert kept_a["maxOutputTokens"] == 999, "local config must survive"
    new_b = next(m for m in endpoint_a["models"] if m["id"] == "model-b")
    # model-b has no source url -> falls back to the target endpoint's existing url
    assert new_b["url"] == "https://api.example.com"
    assert new_b["name"] == "Model B"
    assert new_b["toolCalling"] is True
    assert new_b["vision"] is True
    assert new_b["maxInputTokens"] == 1000000
    assert new_b["maxOutputTokens"] == 384000
    assert new_b["supportsReasoningEffort"] == ["max"]
    assert outcome.changed is True
    assert outcome.providers[0].added == ["model-b"]
    assert outcome.providers[0].kept >= 1


def test_endpoint_fallback_url_used_for_new_models():
    source = _make_source("EndpointA", [{"id": "model-x"}], fallback_url="https://fb.example.com")
    target = [_custom_endpoint("EndpointA", models=[])]
    outcome = sync_config(target, source, allow_delete=True)
    added = outcome.config[0]["models"][0]
    assert added["url"] == "https://fb.example.com"


def test_url_fallback_chain():
    # model-x has no source url; target endpoint has an existing model url.
    source = _make_source("EndpointA", [{"id": "model-x"}])
    target = [_custom_endpoint("EndpointA", models=[_model_obj("old", url="https://old.example.com")])]
    outcome = sync_config(target, source, allow_delete=True)
    added = next(m for m in outcome.config[0]["models"] if m["id"] == "model-x")
    assert added["url"] == "https://old.example.com"


def test_skips_model_when_no_url_anywhere():
    source = _make_source("EndpointA", [{"id": "model-x"}])
    target = [_custom_endpoint("EndpointA", models=[])]
    outcome = sync_config(target, source, allow_delete=True)
    provider = outcome.providers[0]
    assert provider.skipped_model_ids == ["model-x"]
    assert provider.errors, "error recorded for skipped model"
    assert outcome.config[0]["models"] == []


# ---------------------------------------------------------------------------
# 2. existing models keep local config, not overwritten by source
# ---------------------------------------------------------------------------


def test_existing_models_keep_local_config():
    source = _make_source("EndpointA", [
        {"id": "model-a", "name": "Wrong Name", "url": "https://wrong.example.com",
         "maxInputTokens": 12, "maxOutputTokens": 34,
         "supportsReasoningEffort": ["none"], "toolCalling": False, "vision": False},
    ])
    local = _model_obj(
        "model-a",
        name="Local Name",
        url="https://local.example.com",
        maxInputTokens=777,
        maxOutputTokens=888,
        supportsReasoningEffort=["max"],
        toolCalling=True,
        vision=True,
        modelOptions={"top_p": 0.9},
    )
    target = [_custom_endpoint("EndpointA", models=[local])]
    outcome = sync_config(target, source, allow_delete=True)
    kept = outcome.config[0]["models"][0]
    assert kept == local
    assert outcome.providers[0].added == []
    assert outcome.providers[0].removed == []
    assert outcome.changed is False


# ---------------------------------------------------------------------------
# 3. default deletes target-only models and their settings entries
# ---------------------------------------------------------------------------


def test_default_deletes_missing_models_and_settings():
    source = _make_source("EndpointA", [{"id": "keep"}])
    target = [
        _custom_endpoint(
            "EndpointA",
            models=[_model_obj("keep"), _model_obj("gone")],
            settings={"keep": {"reasoningEffort": "max"}, "gone": {"reasoningEffort": "max"}},
        ),
    ]
    outcome = sync_config(target, source, allow_delete=True)
    endpoint_a = outcome.config[0]
    assert [m["id"] for m in endpoint_a["models"]] == ["keep"]
    assert endpoint_a["settings"] == {"keep": {"reasoningEffort": "max"}}
    assert outcome.providers[0].removed == ["gone"]
    assert outcome.providers[0].settings_keys_removed == ["gone"]


def test_settings_field_dropped_when_it_becomes_empty():
    source = _make_source("EndpointA", [{"id": "keep"}])
    target = [
        _custom_endpoint(
            "EndpointA",
            models=[_model_obj("keep"), _model_obj("gone")],
            settings={"gone": {"reasoningEffort": "max"}},
        ),
    ]
    outcome = sync_config(target, source, allow_delete=True)
    assert "settings" not in outcome.config[0]
    assert outcome.providers[0].settings_keys_removed == ["gone"]


# ---------------------------------------------------------------------------
# 4. --no-delete only adds, never deletes / overwrites
# ---------------------------------------------------------------------------


def test_no_delete_keeps_all_and_only_adds():
    source = _make_source("EndpointA", [{"id": "new1"}, {"id": "new2"}])
    target = [
        _custom_endpoint(
            "EndpointA",
            models=[_model_obj("old", name="Old Local")],
            settings={"old": {"reasoningEffort": "max"}},
        ),
    ]
    outcome = sync_config(target, source, allow_delete=False)
    ids = [m["id"] for m in outcome.config[0]["models"]]
    assert "old" in ids and "new1" in ids and "new2" in ids
    assert outcome.config[0]["settings"] == {"old": {"reasoningEffort": "max"}}
    assert outcome.providers[0].removed == []
    assert outcome.providers[0].added == ["new1", "new2"]
    assert outcome.providers[0].skipped_deletion is True


def test_kept_count_semantics_in_both_modes():
    """`kept` = local models that survive the sync.

    allow_delete=True  -> |source ∩ existing| (the rest were deleted)
    allow_delete=False -> |existing|          (nothing is deleted)
    """
    source = _make_source("EndpointA", [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    target = [
        _custom_endpoint(
            "EndpointA",
            models=[_model_obj("a"), _model_obj("b"), _model_obj("stale")],
        ),
    ]
    out_delete = sync_config(target, source, allow_delete=True)
    assert out_delete.providers[0].kept == 2, "delete mode: a + b survive, stale removed"
    assert out_delete.providers[0].removed == ["stale"]

    out_nodel = sync_config(target, source, allow_delete=False)
    assert out_nodel.providers[0].kept == 3, "no-delete mode: all 3 existing kept"
    assert out_nodel.providers[0].removed == []


# ---------------------------------------------------------------------------
# 5. --dry-run writes nothing
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "settings.yaml.imported").write_text(
        "llm-gateway-a:\n  providers:\n    provider-a:\n      displayName: EndpointA\n      models:\n"
        "        - id: model-a\n          url: https://api.example.com\n",
        encoding="utf-8",
    )
    target_path = tmp_path / "chatLanguageModels.json"
    target_path.write_text(serialize([_custom_endpoint("EndpointA", models=[])]) + "\n", encoding="utf-8")
    before = target_path.read_text(encoding="utf-8")

    from dsh_copilot_sync.cli import run

    code = run(["--dsh-dir", str(source_dir), "--config", str(target_path), "--all", "--dry-run"])
    assert code == 0
    assert target_path.read_text(encoding="utf-8") == before, "dry-run must not modify the file"
    assert "Dry run" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 6. source read failure leaves target completely unchanged
# ---------------------------------------------------------------------------


def test_source_read_failure_leaves_target_unchanged(tmp_path):
    missing_dir = tmp_path / "no-such-dsh"
    target_path = tmp_path / "chatLanguageModels.json"
    original = [_custom_endpoint("EndpointA", models=[_model_obj("keep")])]
    target_path.write_text(serialize(original), encoding="utf-8")
    before = target_path.read_text(encoding="utf-8")

    from dsh_copilot_sync.cli import run

    code = run(["--dsh-dir", str(missing_dir), "--config", str(target_path), "--all"])
    assert code == 1, "missing source dir is a configuration error"
    assert target_path.read_text(encoding="utf-8") == before
    assert not any(p.name.startswith(".dsh-sync-") for p in tmp_path.iterdir()), "no temp files left behind"


# ---------------------------------------------------------------------------
# 7. empty source model list skips deletion
# ---------------------------------------------------------------------------


def test_empty_source_model_list_skips_deletion():
    endpoint = DshEndpoint(name="EndpointA", models=[])
    source = DshSourceIndex(endpoints={"EndpointA": endpoint})
    target = [
        _custom_endpoint(
            "EndpointA",
            models=[_model_obj("local1"), _model_obj("local2")],
            settings={"local1": {"reasoningEffort": "max"}},
        ),
    ]
    outcome = sync_config(target, source, allow_delete=True)
    ids = [m["id"] for m in outcome.config[0]["models"]]
    assert ids == ["local1", "local2"], "no deletion when the source has no valid models"
    assert outcome.config[0]["settings"] == {"local1": {"reasoningEffort": "max"}}
    provider = outcome.providers[0]
    assert provider.status == "skipped"
    assert provider.skipped_deletion is True
    assert provider.errors
    assert outcome.changed is False


# ---------------------------------------------------------------------------
# 8. non-customendpoint entries stay unchanged
# ---------------------------------------------------------------------------


def test_non_customendpoint_entries_unchanged():
    vendor_entry = {
        "name": "EndpointA",
        "vendor": "vendor-x",
        "apiKey": "should-not-be-read",
        "models": [_model_obj("never-touched")],
    }
    source = _make_source("EndpointA", [{"id": "new-model"}])
    target = [vendor_entry]
    outcome = sync_config(target, source, allow_delete=True)
    assert outcome.config[0] == vendor_entry, "non-customendpoint entries must pass through untouched"
    # No customendpoint in target -> no provider results at all
    assert outcome.providers == []
    # source endpoint "EndpointA" has no matching customendpoint target
    assert outcome.unmatched_source == ["EndpointA"]
    assert outcome.changed is False


# ---------------------------------------------------------------------------
# 9. vendor == "copilot" entries stay unchanged
# ---------------------------------------------------------------------------


def test_copilot_vendors_unchanged_even_with_same_name():
    copilot_entry = {
        "name": "Copilot",
        "vendor": "copilot",
        "settings": {"auto": {"tier": "intelligence"}},
    }
    copilot_cli_entry = {
        "name": "Copilot",
        "vendor": "agent-host-copilotcli",
        "settings": {"customendpoint/EndpointA/sample-flash": {"thinkingLevel": "max"}},
    }
    source = _make_source("Copilot", [{"id": "injected", "url": "https://x.example.com"}])
    target = [copilot_entry, copilot_cli_entry, _custom_endpoint("EndpointA", models=[])]
    outcome = sync_config(target, source, allow_delete=True)
    assert outcome.config[0] == copilot_entry, "vendor=copilot must pass through byte-identical"
    assert outcome.config[1] == copilot_cli_entry
    # The source endpoint is named "Copilot", but no *customendpoint* in the
    # target is named "Copilot" -> nothing is injected, EndpointA stays untouched.
    assert outcome.config[2]["models"] == []
    assert outcome.changed is False
    # The source endpoint "Copilot" is unmatched (no customendpoint named Copilot).
    assert outcome.unmatched_source == ["Copilot"]


# ---------------------------------------------------------------------------
# 10. duplicate source endpoint names raise
# ---------------------------------------------------------------------------


def test_duplicate_source_endpoint_names_raise(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "a.json").write_text(
        json.dumps([{"name": "EndpointA", "models": [{"id": "m1"}]}]), encoding="utf-8"
    )
    (source_dir / "b.json").write_text(
        json.dumps([{"name": "EndpointA", "models": [{"id": "m2"}]}]), encoding="utf-8"
    )
    with pytest.raises(ModelSyncError, match="duplicate source endpoint name"):
        load_dsh_source(str(source_dir))


def test_duplicate_source_endpoint_names_same_file_raise(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "a.json").write_text(
        json.dumps([
            {"name": "EndpointA", "models": [{"id": "m1"}]},
            {"name": "EndpointA", "models": [{"id": "m2"}]},
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ModelSyncError, match="duplicate source endpoint name"):
        load_dsh_source(str(source_dir))


# ---------------------------------------------------------------------------
# 11. running twice reports no change the second time
# ---------------------------------------------------------------------------


def test_second_run_reports_no_change():
    source = _make_source("EndpointA", [
        {"id": "model-a", "url": "https://api.example.com"},
        {"id": "model-b"},
    ])
    first_target = [
        _custom_endpoint("EndpointA", models=[_model_obj("model-a", maxOutputTokens=999)]),
    ]
    outcome1 = sync_config(first_target, source, allow_delete=True)
    assert outcome1.changed is True

    # Second run starts from the state produced by the first.
    outcome2 = sync_config(outcome1.config, source, allow_delete=True)
    assert outcome2.changed is False
    assert outcome2.config == outcome1.config
    assert all(p.status == "unchanged" for p in outcome2.providers)


# ---------------------------------------------------------------------------
# 12. atomic write skips when content is identical
# ---------------------------------------------------------------------------


def test_atomic_write_skips_when_content_identical(tmp_path):
    data = [_custom_endpoint("EndpointA", models=[_model_obj("a")])]
    path = tmp_path / "cfg.json"
    text = serialize(data)
    path.write_text(text, encoding="utf-8")

    result = write_config_atomic(str(path), data, original_text=read_original_text(str(path)))
    assert result.written is False
    assert "no changes" in result.reason

    # A real write happens when content differs.
    data2 = [
        _custom_endpoint("EndpointA", models=[_model_obj("a"), _model_obj("b")]),
    ]
    result2 = write_config_atomic(str(path), data2, original_text=read_original_text(str(path)))
    assert result2.written is True
    assert load_config(str(path)) == data2


def test_atomic_write_failure_leaves_no_tmp_and_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(serialize([_custom_endpoint("EndpointA", models=[])]) + "\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    import dsh_copilot_sync.config as config_module

    real_replace = __import__("os").replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(config_module.os, "replace", boom)
    from dsh_copilot_sync.models import WriteError

    with pytest.raises(WriteError):
        write_config_atomic(str(path), [_custom_endpoint("EndpointA", models=[_model_obj("a")])])
    assert path.read_text(encoding="utf-8") == before, "original file must survive a failed write"
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".dsh-sync-")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# 13. invalid / non-array target JSON -> config error
# ---------------------------------------------------------------------------


def test_invalid_target_json_is_config_error(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ModelSyncError, match="not valid JSON"):
        load_config(str(path))


def test_non_array_target_is_config_error(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"providers": []}), encoding="utf-8")
    with pytest.raises(ModelSyncError, match="must contain a JSON array"):
        load_config(str(path))


def test_cli_returns_config_error_exit_code(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "a.json").write_text(json.dumps([]), encoding="utf-8")
    target = tmp_path / "cfg.json"
    target.write_text("not json at all", encoding="utf-8")
    from dsh_copilot_sync.cli import run

    code = run(["--dsh-dir", str(source_dir), "--config", str(target), "--all"])
    assert code == 1


# ---------------------------------------------------------------------------
# extra: dsh directory parsing / alias rules
# ---------------------------------------------------------------------------


def test_real_yaml_settings_layout(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "settings.yaml.imported").write_text(
        "llm-gateway-a:\n"
        "  providers:\n"
        "    provider-a:\n"
        "      displayName: EndpointA\n"
        "      apiKeyEnv: ENDPOINT_A_API_KEY\n"
        "      baseURL: https://api.example.com/v1\n"
        "      models:\n"
        "        - id: sample-flash\n"
        "          name: Sample Flash\n"
        "        - id: vendor-a/model-v4-flash\n"
        "    globex:\n"
        "      models:\n"
        "        - id: globex-flash\n"
        "          contextWindow: 1000000\n"
        "          inputModalities: [text, image]\n",
        encoding="utf-8",
    )
    index = load_dsh_source(str(source_dir))
    assert "EndpointA" in index.endpoints
    endpoint_a = index.endpoints["EndpointA"]
    assert [m.id for m in endpoint_a.models] == ["sample-flash", "vendor-a/model-v4-flash"]
    assert endpoint_a.fallback_url == "https://api.example.com"

    assert "Globex" in index.endpoints
    ds = index.endpoints["Globex"]
    flash = ds.find("globex-flash")
    assert flash.maxInputTokens == 1000000
    assert flash.vision is True, "inputModalities: image -> vision"
    assert flash.name is None


def test_plugin_loader_style_yaml(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "plugin-loader.yml").write_text(
        "- id: llm\n"
        "  name: '@corp/dsh-sample'\n"
        "  config:\n"
        "    providers:\n"
        "      provider-b:\n"
        "        displayName: EndpointB\n"
        "        baseURL: https://api.example.com/v1\n"
        "        models:\n"
        "          - id: model-a\n",
        encoding="utf-8",
    )
    index = load_dsh_source(str(source_dir))
    assert "EndpointB" in index.endpoints


def test_normalize_base_url_drops_only_single_trailing_v1():
    from dsh_copilot_sync.dsh_source import normalize_base_url

    # A single trailing /v1 is stripped.
    assert normalize_base_url("https://host/api/v1") == "https://host/api"
    assert normalize_base_url("https://host/v1") == "https://host"
    # Duplicate trailing segments are NOT collapsed (only the last one is removed).
    assert normalize_base_url("https://host/v1/v1") == "https://host/v1"
    # No /v1 -> unchanged (except trailing-slash trim).
    assert normalize_base_url("https://host/api/") == "https://host/api"
    assert normalize_base_url("https://host") == "https://host"
    # Non-string -> None.
    assert normalize_base_url(None) is None
    assert normalize_base_url("") is None
    assert normalize_base_url("   ") is None


def test_model_id_dedup_and_invalid_ids_skipped():
    # Dedup/skip happens during source extraction (_build_endpoint), not in
    # DshModel.from_raw.
    from dsh_copilot_sync.dsh_source import _build_endpoint

    endpoint = _build_endpoint(
        "EndpointA",
        {
            "models": [
                {"id": "a"},
                {"id": "a"},
                {"id": ""},
                {"id": 42},
                "just-a-string",
                None,
                {"id": "b"},
            ]
        },
    )
    ids = [m.id for m in endpoint.models]
    assert ids == ["a", "b"]


def test_duplicate_target_endpoint_name_is_reported_not_fatal():
    source = _make_source("EndpointA", [{"id": "new-model"}])
    target = [
        _custom_endpoint("EndpointA", models=[]),
        _custom_endpoint("EndpointA", models=[]),
    ]
    outcome = sync_config(target, source, allow_delete=True)
    provider = next(p for p in outcome.providers if p.name == "EndpointA")
    assert provider.status == "skipped"
    assert any("appears 2 times" in e for e in provider.errors)
    assert outcome.config[0]["models"] == [] and outcome.config[1]["models"] == []
    assert outcome.changed is False


def test_unknown_provider_flag_raises():
    source = _make_source("EndpointA", [{"id": "new-model"}])
    target = [_custom_endpoint("EndpointA", models=[])]
    with pytest.raises(ModelSyncError, match="unknown target endpoint"):
        sync_config(target, source, allow_delete=True, provider_names=["DoesNotExist"])


def test_missing_target_endpoint_name_is_error():
    source = _make_source("EndpointA", [{"id": "new-model"}])
    target = [_custom_endpoint("EndpointA", models=[])]
    with pytest.raises(ModelSyncError, match="unknown target endpoint"):
        sync_config(target, source, allow_delete=True, provider_names=["endpointa"])


def test_blank_only_provider_name_raises():
    # A --provider value that is only whitespace must not silently act as --all.
    source = _make_source("EndpointA", [{"id": "new-model", "url": "https://x.example.com"}])
    target = [_custom_endpoint("EndpointA", models=[])]
    with pytest.raises(ModelSyncError, match="no usable"):
        sync_config(target, source, allow_delete=True, provider_names=["   "])


def test_new_model_gets_only_canonical_fields():
    source = _make_source("EndpointA", [
        {
            "id": "fresh",
            "name": "Fresh Model",
            "url": "https://api.example.com",
            "object": "model",
            "created": 123,
            "owned_by": "nobody",
        },
    ])
    target = [_custom_endpoint("EndpointA", models=[])]
    outcome = sync_config(target, source, allow_delete=True)
    added = outcome.config[0]["models"][0]
    assert set(added) == {"id", "name", "url", "toolCalling", "vision",
                          "maxInputTokens", "maxOutputTokens", "supportsReasoningEffort"}
    assert "object" not in added and "created" not in added and "owned_by" not in added
    assert added["name"] == "Fresh Model"
    assert added["url"] == "https://api.example.com"


def test_provider_filter_only_touches_selected_endpoint():
    source = _make_source("EndpointA", [{"id": "new-model", "url": "https://x.example.com"}])
    target = [
        _custom_endpoint("EndpointA", models=[]),
        _custom_endpoint("Gamma", models=[]),
    ]
    outcome = sync_config(target, source, allow_delete=True, provider_names=["EndpointA"])
    assert [m["id"] for m in outcome.config[0]["models"]] == ["new-model"]
    assert outcome.config[1]["models"] == [], "Gamma must stay untouched (not selected)"
    assert outcome.changed is True


def test_unmatched_source_endpoints_are_reported():
    source = _make_source("EndpointA", [{"id": "m"}])
    target = [_custom_endpoint("EndpointA", models=[]), _custom_endpoint("Zedco", models=[])]
    outcome = sync_config(target, source, allow_delete=False)
    assert outcome.unmatched_source == []
    # no-source status for target-only endpoint
    zedco = next(p for p in outcome.providers if p.name == "Zedco")
    assert zedco.status == "no-source"


def test_source_without_model_lists_is_config_error(tmp_path):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "random.yml").write_text("hello: world\n", encoding="utf-8")
    from dsh_copilot_sync.models import ModelSyncError

    with pytest.raises(ModelSyncError):
        load_dsh_source(str(source_dir))


def test_missing_source_directory_is_config_error(tmp_path):
    with pytest.raises(ModelSyncError):
        load_dsh_source(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# full CLI round-trip
# ---------------------------------------------------------------------------


def test_cli_round_trip_exit_codes(tmp_path, capsys):
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "a.json").write_text(
        json.dumps([{"name": "EndpointA", "url": "https://api.example.com",
                     "models": [{"id": "model-a"}, {"id": "model-b"}]}]),
        encoding="utf-8",
    )
    target = tmp_path / "cfg.json"
    target.write_text(serialize([_custom_endpoint("EndpointA", models=[_model_obj("model-a")])]), encoding="utf-8")

    from dsh_copilot_sync.cli import run

    code = run(["--dsh-dir", str(source_dir), "--config", str(target), "--all"])
    assert code == 0
    updated = load_config(str(target))
    assert [m["id"] for m in updated[0]["models"]] == ["model-a", "model-b"]

    out = capsys.readouterr().out
    assert "Updated" in out

    # second run: idempotent, no write
    code2 = run(["--dsh-dir", str(source_dir), "--config", str(target), "--all"])
    assert code2 == 0
    out2 = capsys.readouterr().out
    assert "No changes required" in out2


def test_cli_partial_failure_exit_code(tmp_path, capsys):
    # source has a model that cannot be added (no url anywhere) -> endpoint-level error
    source_dir = tmp_path / "dsh"
    source_dir.mkdir()
    (source_dir / "a.json").write_text(
        json.dumps([{"name": "EndpointA", "models": [{"id": "no-url-model"}]}]),
        encoding="utf-8",
    )
    target = tmp_path / "cfg.json"
    target.write_text(serialize([_custom_endpoint("EndpointA", models=[])]), encoding="utf-8")

    from dsh_copilot_sync.cli import run

    code = run(["--dsh-dir", str(source_dir), "--config", str(target), "--all"])
    assert code == 2, f"expected partial-failure exit code, got {code}"
