host=0.0.0.0
port=12345
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port