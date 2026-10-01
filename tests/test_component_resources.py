from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest
from jester_core import ComponentManifest, component_identity
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


PACKAGE = Path(__file__).parents[1]
RESOURCE_ROOT = PACKAGE / "src" / "mixed_inference" / "resources"
HASH_A, HASH_B, HASH_C = "a" * 64, "b" * 64, "c" * 64


def _json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _payload() -> dict:
    return {
        "baseModel": {
            "asset": {"id": "model.example", "version": "1", "sha256": HASH_A},
            "artifactFiles": {"id": "model.files", "version": "1", "sha256": HASH_B},
            "artifactSha256": HASH_C,
            "canonicalId": f"sha256:{HASH_C}",
            "revision": "revision-1",
            "layerPath": "model.layers",
            "dtype": "bfloat16",
        },
        "facetRegistry": {"asset": {"id": "facets.example", "version": "1", "sha256": HASH_C}},
        "workers": 1,
        "maxRequestBytes": 2 * 1024 * 1024,
    }


def test_manifest_binds_exact_resources_and_dedicated_capabilities() -> None:
    manifest = ComponentManifest.model_validate(_json(RESOURCE_ROOT / "component-manifest.json"))
    assert (manifest.id, manifest.version) == ("mi", "0.1.0")
    assert component_identity(manifest).manifest_sha256 == manifest.semantic_sha256
    assert {item.capability.id for item in manifest.capabilities} == {"mi.generate", "mi.plan"}
    assert all(item.capability.concurrency.sharing == "dedicated-only" for item in manifest.capabilities)
    assert all(item.capability.concurrency.max_concurrent_requests == 1 for item in manifest.capabilities)
    for resource in manifest.resources:
        path = RESOURCE_ROOT.parent / resource.installed_path
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256


def test_schema_is_strict_portable_and_identity_equality_is_not_claimed() -> None:
    validator = Draft202012Validator(_json(RESOURCE_ROOT / "mi-serve-configuration-v1.schema.json"))
    validator.validate(_payload())
    assert _json(RESOURCE_ROOT / "mi-serve-configuration-defaults-v1.json") == {
        "workers": 1, "maxRequestBytes": 2 * 1024 * 1024
    }
    invalid = []
    for extra in ({"hostId": "host.one"}, {"device": "cuda:0"}, {"leaseId": "lease.one"}, {"online": True}):
        invalid.append(_payload() | extra)
    for field, value in (("dtype", "auto"), ("layerPath", "../model"), ("revision", "rev\n")):
        item = _payload(); item["baseModel"][field] = value; invalid.append(item)
    for field in ("asset", "artifactFiles"):
        item = _payload(); item["baseModel"][field]["sha256"] += "\n"; invalid.append(item)
    item = _payload(); item["workers"] = 2; invalid.append(item)
    item = _payload(); item["maxRequestBytes"] = 2097153; invalid.append(item)
    for item in invalid:
        with pytest.raises(ValidationError):
            validator.validate(item)

    # Structural schema deliberately cannot establish MI's identity equality.
    unequal = _payload(); unequal["baseModel"]["canonicalId"] = f"sha256:{HASH_A}"
    validator.validate(unequal)
    # JSON Schema treats mathematically integral 1.0 as an integer; lexical
    # canonicalization belongs to the future mapping consumer.
    integral_float = _payload(); integral_float["workers"] = 1.0
    validator.validate(integral_float)


def test_mapping_is_closed_and_every_flag_exists_on_real_serve_parser() -> None:
    mapping = _json(RESOURCE_ROOT / "mi-serve-adapter-mapping-v1.json")
    flags = [item["flag"] for item in mapping["arguments"]]
    assert flags == [
        "--base-model", "--base-identity", "--revision", "--layer-path", "--dtype",
        "--artifact-files", "--artifact-sha256", "--registry", "--workers", "--max-request-bytes",
    ]
    assert mapping["environment"] == {} and mapping["additionalArguments"] is False
    assert not {"--device", "--host", "--port"}.intersection(flags)

    tree = ast.parse((PACKAGE / "src" / "mixed_inference" / "args.py").read_text(encoding="utf-8"))
    parser_flags = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for arg in node.args[:1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
    }
    assert set(flags).issubset(parser_flags)
    workers_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--workers"
    )
    workers_keywords = {item.arg: item.value for item in workers_call.keywords if item.arg}
    assert isinstance(workers_keywords["type"], ast.Name) and workers_keywords["type"].id == "int"


def test_capability_paths_match_managed_server_authorization_source() -> None:
    contracts = [_json(RESOURCE_ROOT / name) for name in ("mi-generate-capability-v1.json", "mi-plan-capability-v1.json")]
    source = (PACKAGE / "src" / "mixed_inference" / "api" / "server.py").read_text(encoding="utf-8")
    for contract in contracts:
        assert contract["method"] == "POST"
        assert f'"{contract["paths"][0]}": "{contract["channelOperation"]}"' in source


@pytest.mark.skipif(os.environ.get("JESTER_MI_INSTALLED_RESOURCE_PROOF") != "1", reason="isolated wheel proof only")
def test_installed_resources_without_importing_mi_or_torch() -> None:
    assert "mixed_inference" not in sys.modules and "torch" not in sys.modules
    distribution = metadata.distribution("jester-mixed-inference")
    package_root = Path(distribution.locate_file("mixed_inference")).resolve()
    manifest = ComponentManifest.model_validate(_json(package_root / "resources" / "component-manifest.json"))
    assert "mixed_inference" not in sys.modules and "torch" not in sys.modules
    installed = {str(item).replace("\\", "/") for item in distribution.files or ()}
    for resource in manifest.resources:
        relative = f"mixed_inference/{resource.installed_path}"
        assert relative in installed
        path = Path(distribution.locate_file(relative)).resolve()
        assert path.is_relative_to(package_root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource.sha256
    assert "mixed_inference" not in sys.modules and "torch" not in sys.modules
