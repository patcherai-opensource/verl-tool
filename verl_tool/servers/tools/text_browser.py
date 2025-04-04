# from .base import BaseTool, register_tool, registered_tools
#
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# import sqlite3
# import os
# import pickle
# from filelock import FileLock
#
#
# class ObjectStore:
#     def __init__(self, db_path='objects.db'):
#         """
#         :param db_path: Path to the database file used for storage
#         """
#         self.db_path = db_path
#         self._lockfile = self.db_path + ".lock"
#         self._init_db()
#
#     def _init_db(self):
#         """Initialize the database and create the table if it doesn't exist."""
#         with FileLock(self._lockfile):
#             conn = sqlite3.connect(self.db_path)
#             c = conn.cursor()
#             c.execute('''
#                 CREATE TABLE IF NOT EXISTS objects (
#                     uuid TEXT PRIMARY KEY,
#                     data BLOB
#                 )
#             ''')
#             conn.commit()
#             conn.close()
#
#     def add_object(self, uuid_str, obj):
#         """
#         Store (or update) a Python object in the database.
#         :param uuid_str: UUID string used as the primary key
#         :param obj: Python object to store; it will be serialized using pickle
#         """
#         data_blob = pickle.dumps(obj)
#         with FileLock(self._lockfile):
#             conn = sqlite3.connect(self.db_path)
#             c = conn.cursor()
#             c.execute('''
#                 INSERT OR REPLACE INTO objects (uuid, data) VALUES (?, ?)
#             ''', (uuid_str, data_blob))
#             conn.commit()
#             conn.close()
#
#     def get_object(self, uuid_str):
#         """
#         Retrieve a Python object from the database by its UUID.
#         :param uuid_str: UUID string
#         :return: Deserialized Python object if found, otherwise None
#         """
#         with FileLock(self._lockfile):
#             conn = sqlite3.connect(self.db_path)
#             c = conn.cursor()
#             c.execute('SELECT data FROM objects WHERE uuid = ?', (uuid_str,))
#             row = c.fetchone()
#             conn.close()
#             if row:
#                 return pickle.loads(row[0])
#             return None
#
#     def delete_object(self, uuid_str):
#         """
#         Delete an object from the database by its UUID.
#         :param uuid_str: UUID string
#         :return: True if the object was deleted, False if not found
#         """
#         with FileLock(self._lockfile):
#             conn = sqlite3.connect(self.db_path)
#             c = conn.cursor()
#             c.execute('DELETE FROM objects WHERE uuid = ?', (uuid_str,))
#             rowcount = c.rowcount
#             conn.commit()
#             conn.close()
#             return rowcount > 0
#
#
# @register_tool
# class TextBrowserTool(BaseTool):
#     tool_type = "text_browser"
#
#     def get_usage_inst(self):
#         return ("Usage instructions for TextBrowser. This code is based on mini_webarena, using playwright to get "
#                 "accessibility tree for LLMs agent easier access. The code is modified from AutoWebGLM. To get start, run `pip install -e .` under the mini_webarena repo.")
#
#     def __init__(self, num_workers=1, store_path='env_store.db'):
#         self.num_workers = num_workers
#         registered_tools[self.tool_type] = self.__class__
#         self.object_store = ObjectStore(store_path)
#
#     def load_env(self, trajectory_id):
#         """
#         Load the environment for the given trajectory_id from the object store.
#         If not found, create a new environment.
#         """
#         env = self.object_store.get_object(trajectory_id)
#         if env is None:
#             env = {
#                 "trajectory_id": trajectory_id,
#                 "metadata": {
#                     "turns": 0,
#                 },
#                 "previous_obs": [],
#             }
#         return env
#
#     def save_env(self, trajectory_id, env):
#         """
#         Save the environment for the given trajectory_id to the object store.
#         """
#         self.object_store.add_object(trajectory_id, env)
#
#     def delete_env(self, trajectory_id):
#         """
#         Delete the environment for the given trajectory_id from the object store.
#         """
#         self.object_store.delete_object(trajectory_id)
#
#     def conduct_action(self, trajectory_id, action, extra_field):
#         # extra_fields: {question: str or None, gt: str or None, url: str or None}
#         # print(trajectory_id, action, extra_field)
#         from mini_webarena.env_worker import WikiQAEnv
#         import copy
#         print("#### extra_field ####", extra_field)
#         extra_field = extra_field['extra_fields']
#         env_state = self.load_env(trajectory_id)
#         if env_state.get("trajectory_id") is not None: # New Environment, need start
#             question = extra_field['question'] if extra_field['question'] is not None else "placeholder"
#             gt = extra_field['gt'] if extra_field['gt'] is not None else "placeholder"
#             url = extra_field['url']
#             env = WikiQAEnv(question, gt, url = url, prompt_format = "full")
#             env_state = copy.deepcopy(env.get_state())
#             self.save_env(trajectory_id, env_state)
#             if action is None:
#                 observation = env.render()
#                 env.close()
#                 return observation, False, True
#             env.close()
#             del env
#         env = WikiQAEnv(env_state["question"], env_state["gt"], url=env_state["url"], prompt_format = "full")
#         env.load_state(env_state)
#         observation, _, done, _ = env.step(action)
#         if done:
#             self.delete_env(trajectory_id)
#         else:
#             env_state = copy.deepcopy(env.get_state())
#             self.save_env(trajectory_id, env_state)
#         print("#### observation before save ####")
#         print(env.env._get_obs())
#         env.close()
#         # Regenerate the observation
#         env = WikiQAEnv(env_state["question"], env_state["gt"], url=env_state["url"], prompt_format = "full")
#         env.load_state(env_state)
#         print("#### observation after save ####")
#         print(env.env._get_obs())
#         env.close()
#         return observation, done, True
#
#     def get_observations(self, trajectory_ids, actions, extra_fields):
#         # with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#         with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#             results = list(
#                 tqdm(executor.map(self.conduct_action, trajectory_ids, actions, extra_fields),
#                      total=len(trajectory_ids), desc=f"Getting observations using tool {self.tool_type}"))
#
#         observations, dones, valids = zip(*results)
#         return observations, dones, valids


import ray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base import BaseTool, register_tool, registered_tools
# 请根据实际路径调整 WikiQAEnv 的导入方式
from mini_webarena.env_worker import WikiQAEnv


@ray.remote
class WikiEnvActor:
    """
    Ray Actor，用于管理一个 WikiQAEnv 实例。
    """

    def __init__(self, question: str, gt: str, url: str = None):
        # prompt_format 设置为 "full" 以保持与原逻辑一致
        self.env = WikiQAEnv(question, gt, url=url, prompt_format="full")

    def start_env(self) -> str:
        """
        初次渲染，返回初始 observation。
        """
        obs = self.env.render()
        return obs

    def step_env(self, query: str) -> (str, int):
        """
        对环境执行一步操作。
        :param query: 用户的操作/查询指令
        :return: (observation, done_flag)，done_flag 为 1 表示环境结束，0 表示未结束。
        """
        obs, reward, done, info = self.env.step(query)
        if done:
            self.env.close()
        return obs, 1 if done else 0


@register_tool
class TextBrowserTool(BaseTool):
    tool_type = "text_browser"

    def get_usage_inst(self):
        return ("TextBrowserTool 使用 Ray 调度器管理 WikiQAEnv 环境，"
                "每个 trajectory_id 对应一个 Ray Actor，支持初次渲染（action=None）和 step 操作。")

    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        # 保存 trajectory_id -> Ray Actor 的映射
        self.env_actors = {}
        # 记录 Actor 创建顺序，用于超量时清理最早创建的 Actor
        self.actor_creation_order = []

    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        根据 trajectory_ids 和 actions 获取环境 observation。
        对于未创建环境的 trajectory_id，根据 extra_fields 中的信息创建新的 Ray Actor，
        如果 action 为 None，则调用 start_env 得到初次渲染，否则调用 step_env。

        :param trajectory_ids: list，每个环境的唯一标识
        :param actions: list，每个环境对应的 action（当为 None 时认为是初次调用）
        :param extra_fields: list，每个元素为一个字典，包含 keys 如 "question"、"gt"、"url"
                             若字典中包含 "extra_fields" 键，则取其对应的子字典
        :return: (observations, dones, valid_flags)
                 observations: list of observation 字符串
                 dones: list，每个环境的状态（0: 未结束，1: 结束，-1: 无效）
                 valid_flags: list，标记是否为有效调用（全部 True）
        """
        futures = []
        for i, trajectory_id in enumerate(trajectory_ids):
            action = actions[i]
            # 若 extra_fields 中包含嵌套的 "extra_fields"，则提取之
            extra = extra_fields[i].get("extra_fields", extra_fields[i])
            if trajectory_id not in self.env_actors:
                # 根据 extra_fields 中的参数创建新的 Actor
                question = extra.get("question", "placeholder")
                gt = extra.get("gt", "placeholder")
                url = extra.get("url")
                actor = WikiEnvActor.remote(question, gt, url)
                self.env_actors[trajectory_id] = actor
                self.actor_creation_order.append(trajectory_id)
                # 如果 action 为 None，则调用初始渲染，否则直接调用 step 操作
                if action is None:
                    fut = actor.start_env.remote()
                else:
                    fut = actor.step_env.remote(action)
                futures.append((trajectory_id, fut))
            else:
                # 已存在对应 Actor，直接调用对应接口
                actor = self.env_actors[trajectory_id]
                if action is None:
                    fut = actor.start_env.remote()
                else:
                    fut = actor.step_env.remote(action)
                futures.append((trajectory_id, fut))
            # 超过 1000 个 Actor 时清理最早创建的 Actor
            while len(self.env_actors) > 50:
                oldest = self.actor_creation_order.pop(0)
                if oldest in self.env_actors:
                    ray.kill(self.env_actors[oldest], no_restart=True)
                    del self.env_actors[oldest]

        # 收集所有结果
        observations = [None] * len(trajectory_ids)
        dones = [None] * len(trajectory_ids)
        valid_flags = [True] * len(trajectory_ids)
        for i, (trajectory_id, fut) in enumerate(futures):
            result = ray.get(fut)
            if isinstance(result, tuple):
                obs, done = result
            else:
                obs = result
                done = 0
            observations[i] = obs
            dones[i] = done
            # 如果环境已结束，删除对应的 Actor
            if done == 1:
                if trajectory_id in self.env_actors:
                    ray.kill(self.env_actors[trajectory_id], no_restart=True)
                    del self.env_actors[trajectory_id]
                if trajectory_id in self.actor_creation_order:
                    self.actor_creation_order.remove(trajectory_id)
        return observations, dones, valid_flags
