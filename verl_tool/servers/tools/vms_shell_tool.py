from .base import BaseTool, register_tool
import docker
import asyncio
import json
from lib.llm_tools.shell_tool import ShellTool, ShellToolArgs
from lib.llm_tools.tools import find_xml_blocks
from vms.framework.types import RepositoryReference, RunPaths
from vms.framework.git_manager import default_git_manager_factory
from lib.execenv.execenv import ExecEnvParameters, DockerExecEnv, start_container, setup_execenv

@register_tool
class VmsShellTool(BaseTool):
    tool_type = "vms_shell_tool"
    stop_tokens = [] # TODO: ["```output", "<o>", "<tool_call>"]
    # TODO: make configurable
    command_timeout_seconds = 60

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.sessions = {}  # trajectory_id -> {"execenv": ..., "run_paths": ..., "shell_tool": ...}

    def get_usage_inst(self):
        # NOTE: this is not used
        return (
            "You are able to execute shell commands in a secure Docker container. "
            "Use <shell_tool>...</shell_tool> blocks to specify commands."
        )

    def has_env(self, trajectory_id):
        return trajectory_id in self.env_cache

    def load_env(self, trajectory_id):
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {"turns": 0},
                "previous_obs": [],
                "session_active": False,
            }
        return env

    def save_env(self, trajectory_id, env):
        self.env_cache[trajectory_id] = env

    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })

    def delete_env(self, trajectory_id):
        session = self.sessions.get(trajectory_id)
        if session:
            # Clean up execenv (stop/remove container)
            execenv = session.get("execenv")
            if execenv is not None:
                try:
                    # Stop and remove the container
                    container = execenv.container
                    container.kill()
                    container.remove()
                except Exception:
                    pass
            del self.sessions[trajectory_id]
        if trajectory_id in self.env_cache:
            del self.env_cache[trajectory_id]

    def parse_action(self, action: str):
        # Only look for a single <shell_tool>...</shell_tool> block
        match = next(find_xml_blocks(action, "shell_tool"), None)
        if not match:
            return "", False
        parsed_command = match.group(1).strip()
        return parsed_command, True

    async def _get_or_create_session(self, trajectory_id, extra_field):
        if trajectory_id in self.sessions:
            return self.sessions[trajectory_id]
        # 1. Deserialize RepositoryReference from extra_field
        repo_ref_json = extra_field.get("repository_reference")
        if isinstance(repo_ref_json, str):
            # FIXME: only use one of these, just not sure if it's a string or a dict yet
            repo_ref_dict = json.loads(repo_ref_json)
        else:
            repo_ref_dict = repo_ref_json
        repo_ref = RepositoryReference(**repo_ref_dict)
        # 2. Create RunPaths
        run_paths = RunPaths.new(repo_ref)
        # 3. Clone/update repo
        git_manager = default_git_manager_factory().new(repo_ref)
        git_manager.clone_or_update_repo()
        # 4. Create ExecEnvParameters
        params = ExecEnvParameters(
            image_name="vulnerability-discovery-default",
            repo_dir=run_paths.root_dir,
            raise_on_nonzero=False,
            save_modified_files=False,
        )
        # 5. Start execenv (container)
        docker_client = None
        execenv = None
        try:
            docker_client = setup_execenv(docker.from_env())
            container = start_container(docker_client, params)
            execenv = DockerExecEnv(
                docker_client=docker_client,
                params=params,
                container=container,
                workdir="/volumes/workspace",
            )
            # Copy repo to workspace (as in new_exec_env)
            await execenv.copy_repo_to_workspace()
        except Exception as e:
            raise RuntimeError(f"Failed to start execenv: {e}")
        # 6. Create ShellTool
        shell_tool = ShellTool(
            exec_env=execenv,
            command_timeout_seconds=self.command_timeout_seconds,
            container_kill_on_timeout_seconds=None,
        )
        session = {"execenv": execenv, "run_paths": run_paths, "shell_tool": shell_tool}
        self.sessions[trajectory_id] = session
        return session

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        if not is_valid or not parsed_action.strip():
            observation = ""
            execution_result = ""
            valid = False
            done = False
        else:
            # Ensure session/execenv is ready
            try:
                session = asyncio.run(self._get_or_create_session(trajectory_id, extra_field))
                shell_tool: ShellTool = session["shell_tool"]
                result = asyncio.run(shell_tool.run(ShellToolArgs(command=parsed_action)))
                execution_result = result
                observation = result
                valid = True
                done = False
            except Exception as e:
                execution_result = f"ShellTool error: {str(e)}"
                observation = execution_result
                valid = False
                done = False
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        return observation, done, valid
