from .base import BaseTool, register_tool, registered_tools

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sqlite3
import os
import pickle
from filelock import FileLock


class ObjectStore:
    def __init__(self, db_path='objects.db'):
        """
        :param db_path: Path to the database file used for storage
        """
        self.db_path = db_path
        self._lockfile = self.db_path + ".lock" 
        self._init_db()

    def _init_db(self):
        """Initialize the database and create the table if it doesn't exist."""
        with FileLock(self._lockfile):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS objects (
                    uuid TEXT PRIMARY KEY,
                    data BLOB
                )
            ''')
            conn.commit()
            conn.close()

    def add_object(self, uuid_str, obj):
        """
        Store (or update) a Python object in the database.
        :param uuid_str: UUID string used as the primary key
        :param obj: Python object to store; it will be serialized using pickle
        """
        data_blob = pickle.dumps(obj)
        with FileLock(self._lockfile):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO objects (uuid, data) VALUES (?, ?)
            ''', (uuid_str, data_blob))
            conn.commit()
            conn.close()

    def get_object(self, uuid_str):
        """
        Retrieve a Python object from the database by its UUID.
        :param uuid_str: UUID string
        :return: Deserialized Python object if found, otherwise None
        """
        with FileLock(self._lockfile):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT data FROM objects WHERE uuid = ?', (uuid_str,))
            row = c.fetchone()
            conn.close()
            if row:
                return pickle.loads(row[0])
            return None

    def delete_object(self, uuid_str):
        """
        Delete an object from the database by its UUID.
        :param uuid_str: UUID string
        :return: True if the object was deleted, False if not found
        """
        with FileLock(self._lockfile):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM objects WHERE uuid = ?', (uuid_str,))
            rowcount = c.rowcount
            conn.commit()
            conn.close()
            return rowcount > 0


@register_tool
class TextBrowserTool(BaseTool):
    tool_type = "text_browser"

    def get_usage_inst(self):
        return ("Usage instructions for TextBrowser. This code is based on mini_webarena, using playwright to get "
                "accessibility tree for LLMs agent easier access. The code is modified from AutoWebGLM. To get start, run `pip install -e .` under the mini_webarena repo.")

    def __init__(self, num_workers=16, store_path='env_store.db'):
        self.num_workers = num_workers
        registered_tools[self.tool_type] = self.__class__
        self.object_store = ObjectStore(store_path)

    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id from the object store.
        If not found, create a new environment.
        """
        env = self.object_store.get_object(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
            }
        return env

    def save_env(self, trajectory_id, env):
        """
        Save the environment for the given trajectory_id to the object store.
        """
        self.object_store.add_object(trajectory_id, env)

    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id from the object store.
        """
        self.object_store.delete_object(trajectory_id)

    def conduct_action(self, trajectory_id, action, extra_field):
        # extra_fields: {question: str or None, gt: str or None, url: str or None}
        # print(trajectory_id, action, extra_field)
        from mini_webarena.env_worker import WikiQAEnv
        import copy
        print("#### extra_field ####", extra_field)
        extra_field = extra_field['extra_fields']
        env_state = self.load_env(trajectory_id)
        if env_state.get("trajectory_id") is not None: # New Environment, need start
            question = extra_field['question'] if extra_field['question'] is not None else "placeholder"
            gt = extra_field['gt'] if extra_field['gt'] is not None else "placeholder"
            url = extra_field['url']
            env = WikiQAEnv(question, gt, url = url, prompt_format = "full")
            env_state = copy.deepcopy(env.get_state())
            self.save_env(trajectory_id, env_state)
            if action is None:
                observation = env.render()
                env.close()
                return observation, False, True
            env.close()
            del env
        env = WikiQAEnv(env_state["question"], env_state["gt"], url=env_state["url"], prompt_format = "full")
        env.load_state(env_state)
        observation, _, done, _ = env.step(action)
        if done:
            self.delete_env(trajectory_id)
        else:
            env_state = copy.deepcopy(env.get_state())
            self.save_env(trajectory_id, env_state)
        env.close()
        return observation, done, True

    def get_observations(self, trajectory_ids, actions, extra_fields):
        # with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(executor.map(self.conduct_action, trajectory_ids, actions, extra_fields),
                     total=len(trajectory_ids), desc=f"Getting observations using tool {self.tool_type}"))

        observations, dones, valids = zip(*results)
        return observations, dones, valids
