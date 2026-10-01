from __future__ import annotations

import os
import pytest

if os.environ.get("JESTER_CROSS_REPO_TESTS") != "1":
    pytest.skip("cross-repo RM/component integration; set JESTER_CROSS_REPO_TESTS=1", allow_module_level=True)

import json
import os
import threading
import sysconfig
from importlib.metadata import distribution
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import jester_rm
from jester_rm.config import Settings
from jester_rm.controller.service import ControllerService
from jester_rm.models import Deployment, EntityMetadata, HostSpec, Lease

from mixed_inference.channel_sessions import ManagedChannelContext, ManagedChannelGuard
from mixed_inference.api.server import MixedInferenceHTTPServer


def _assert_installed_rm_origin() -> None:
    module_file = Path(jester_rm.__file__).resolve()
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    old_rm_checkout = Path(__file__).resolve().parents[3] / "packages" / "rm"
    assert module_file.is_relative_to(site_packages), module_file
    assert not module_file.is_relative_to(old_rm_checkout), module_file
    installed = distribution("jester-resource-manager")
    files = tuple(installed.files or ())
    module_record = next(entry for entry in files if str(entry).replace("\\", "/") == "jester_rm/__init__.py")
    record = next(entry for entry in files if entry.name == "RECORD")
    assert Path(installed.locate_file(module_record)).resolve() == module_file
    assert Path(installed.locate_file(record)).resolve().is_relative_to(site_packages)


_assert_installed_rm_origin()


class _RouteService:
    def __init__(self):
        import threading
        self.resource_lock = threading.RLock()
        self.runtime = type("Runtime", (), {"plan": lambda _self, _facets: {"accepted": "plan"}, "generate": lambda _self, _payload: {"accepted": "generate"}})()
    def health(self): return {"status": "ok"}


def _post(port: int, credential: str) -> tuple[int, dict]:
    request = Request(f"http://127.0.0.1:{port}/plan", data=b'{}', method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("X-Jester-Component-Session", credential)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_rm_issued_session_controls_actual_mi_plan_handler(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JESTER_RM_HOME", str(tmp_path / "rm-home"))
    monkeypatch.setenv("JESTER_RM_AGENT_HOME", str(tmp_path / "rm-agent"))
    controller = ControllerService(Settings.load(create=True))
    server = None
    thread = None
    try:
        host = HostSpec(metadata=EntityMetadata(id="host-one"), transport="local", status="online")
        controller.put_host(host)
        from jester_rm.models import ComponentSpec, InstanceBody, InstanceSpec, ResourceRequirements
        def instance(identifier: str, adapter: str, port: int) -> InstanceSpec:
            return InstanceSpec(metadata=EntityMetadata(id=identifier), spec=InstanceBody(
                hostId="host-one", component=ComponentSpec(adapter=adapter), resources=ResourceRequirements.model_validate({"cpu":{},"memory":{"ramBytes":1},"storage":{"poolId":"pool-one","quotaBytes":1}}), network={"port": port},
            ))
        for item in (instance("engine-one", "engine.serve", 8780), instance("mi-one", "mi.serve", 9123)):
            controller.store.put_entity("instance", item.metadata.id, item.api_dict(), expected_generation=0)
        deployment = Deployment.model_validate({"metadata":{"id":"deployment-one"}, "spec":{
            "projectId":"project-one", "engineInstanceId":"engine-one", "members":[{"memberId":"mi","instanceId":"mi-one"}],
            "controllerChannels":[{"channelId":"engine-mi","destinationMemberId":"mi","destinationPort":"http","purpose":"inference","allowedOperations":["mi.plan"]}],
        }})
        stored = controller.put_deployment(deployment)
        running = Deployment.model_validate(stored).model_copy(update={"state":"running"})
        running = Deployment.model_validate(controller.store.put_entity("deployment", "deployment-one", running.api_dict(), expected_generation=1))
        for lease_id, instance_id, port in (("engine-lease", "engine-one", 8780), ("mi-lease", "mi-one", 9123)):
            controller.store.put_lease(Lease(leaseId=lease_id, instanceId=instance_id, instanceGeneration=1, hostId="host-one", hostEpoch="epoch", state="active", configSha256="hash", telemetryToken="token", paths={"root":str(tmp_path / instance_id)}, ports={"http":port}).api_dict())
        grant = controller._issue_engine_grant(running)
        first = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        def local(operation: str, method: str):
            return {"destinationInstanceId":"mi-one","destinationInstanceGeneration":1,"destinationLeaseId":"mi-lease","operation":operation,"method":method}
        guard = ManagedChannelGuard(ManagedChannelContext(rm_url="http://unused"), introspect=lambda credential, expected: controller.introspect_controller_channel_session(credential, expectation=expected) or {}, expectation_provider=local)
        server = MixedInferenceHTTPServer(("127.0.0.1", 0), _RouteService(), managed_channel_guard=guard, require_managed_session=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        assert _post(server.server_port, first["credential"]) == (200, {"accepted":"plan"})
        renewed = controller.controller_channel_sessions(grant["credential"], project_id="project-one", deployment_id="deployment-one")["items"][0]
        assert _post(server.server_port, first["credential"])[0] == 401
        assert _post(server.server_port, renewed["credential"])[0] == 200
        guard._expectation_provider = lambda operation, method: {**local(operation, method), "destinationLeaseId":"wrong-lease"}
        assert _post(server.server_port, renewed["credential"])[0] == 401
    finally:
        if server is not None:
            server.shutdown(); server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        controller.close()
        if thread is not None:
            assert not thread.is_alive()

