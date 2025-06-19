import pytest
from verl_tool.servers.tools.vms_shell_tool import VmsShellTool
import requests

EXAMPLE_REPO_REF = {
    "type": "github",
    "repo": "canvg/canvg",
    "ref": "937668eced93e0335c67a255d0d2277ea708b2cb",
    "directory": ""
}

"""
Integration test usage:
- Start the tool server in another terminal:
    python -m verl_tool.servers.serve --tool_type vms_shell_tool --port 5500
- Then run this test file with pytest:
    pytest verl_tool/servers/tests/test_vms_shell_tool.py
"""

def test_shell_tool_tree():
    tool = VmsShellTool()
    trajectory_id = "test-trajectory"
    action = "<shell_tool>tree -L 2</shell_tool>"
    extra_field = {"repository_reference": EXAMPLE_REPO_REF}
    try:
        obs, done, valid = tool.conduct_action(trajectory_id, action, extra_field)
        assert "README.md" in obs
        assert valid
        assert not done

        action2 = "<shell_tool>cat README.md</shell_tool>"
        obs2, done2, valid2 = tool.conduct_action(trajectory_id, action2, extra_field)
        assert "canvg" in obs2
        assert valid2
        assert not done2
    finally:
        tool.delete_env(trajectory_id)
        

def test_shell_tool_filesystem_statefulness():
    tool = VmsShellTool()
    trajectory_id = "test-fs"
    extra_field = {"repository_reference": EXAMPLE_REPO_REF}
    try:
        # Write to a file
        action1 = "<shell_tool>echo bar > /tmp/testfile.txt</shell_tool>"
        obs1, done1, valid1 = tool.conduct_action(trajectory_id, action1, extra_field)
        assert valid1
        # Read from the file in a new call
        action2 = "<shell_tool>cat /tmp/testfile.txt</shell_tool>"
        obs2, done2, valid2 = tool.conduct_action(trajectory_id, action2, extra_field)
        assert "bar" in obs2
        assert valid2
        assert not done2
    finally:
        tool.delete_env(trajectory_id)

def test_shell_tool_server_tree():
    url = "http://localhost:5500/get_observation"
    health_url = "http://localhost:5500/health"
    try:
        health_resp = requests.get(health_url, timeout=3)
        if health_resp.status_code != 200 or health_resp.json().get("status") != "healthy":
            pytest.skip("Tool server /health endpoint did not return healthy")
    except requests.RequestException:
        pytest.skip("Tool server is not running on localhost:5500 (health check failed)")

    data1 = {
        "trajectory_ids": ["test-trajectory"],
        "actions": ["<shell_tool>tree -L 2</shell_tool>"],
        "extra_fields": [{"repository_reference": EXAMPLE_REPO_REF}]
    }
    resp1 = requests.post(url, json=data1, timeout=10)
    assert resp1.status_code == 200
    result1 = resp1.json()
    assert "README.md" in result1["observations"][0]
    assert result1["valids"][0]
    assert not result1["dones"][0]

    data2 = {
        "trajectory_ids": ["test-trajectory"],
        "actions": ["<shell_tool>cat README.md</shell_tool>"],
        "extra_fields": [{"repository_reference": EXAMPLE_REPO_REF}]
    }
    resp2 = requests.post(url, json=data2, timeout=10)
    assert resp2.status_code == 200
    result2 = resp2.json()
    assert "canvg" in result2["observations"][0]
    assert result2["valids"][0]
    assert not result2["dones"][0]

if __name__ == "__main__":
    test_shell_tool_tree()

"""
curl -X POST http://localhost:5500/get_observation \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory_ids": ["test-trajectory"],
    "actions": ["<shell_tool>tree -L 2</shell_tool>"],
    "extra_fields": [{"repository_reference": {"repo": "canvg/canvg", "ref": "937668eced93e0335c67a255d0d2277ea708b2cb", "directory": ""}}]
  }'
"""
