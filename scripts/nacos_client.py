from modules import devices, script_callbacks, shared
from fastapi import FastAPI, Body
import gradio as gr
import numpy as np
from nacos_client.client import NacosClient
import asyncio
from ia_logging import ia_logging
import socket
import sys
from ia_config import IAConfig, get_ia_config
def get_local_ip():
    ip = socket.gethostbyname(socket.gethostname())
    return ip

server_address = 'http://120.78.215.48:8848'
namespace = '3bcb0b1d-1c2b-4069-97df-9612dac5005c'
username = ''  # 可选，访问密钥
password = ''  # 可选，访问密钥

service_name = 'sd-service'  # 注册的服务名
ip = get_local_ip()  # 本机IP地址
port = 7861  # 服务端口号
weight = 1.0  # 权重，默认为1.0
cluster_name = 'dev'  # 可选，集群名，默认为'default'
metadata = {}  # 可选，元数据，用于自定义信息
def set_nacos_config():
    global server_address, namespace, username, password
    profiles = 'DEFAULT'
    # ia_logging.info(sys.argv)
    for idx in range(len(sys.argv)):
        arg = sys.argv[idx]
        if arg.startswith('--profiles'):
            profiles = sys.argv[idx + 1].upper()
            break
    ia_logging.info(f'current profiles is {profiles}')
    server_address = get_ia_config('nacos_server', profiles)
    namespace = get_ia_config('nacos_namespace', profiles)
    username = get_ia_config('nacos_username', profiles)
    password = get_ia_config('nacos_password', profiles)

def nacos_client_connect(_: gr.Blocks, app: FastAPI):
    global client
    set_nacos_config()
    ia_logging.info(f'nacos client start connecting:{server_address},{namespace}')
    client = NacosClient(server_addresses=server_address, namespace=namespace, password=f"{password}", username=f"{username}")
    nacos_conn_status = client.add_naming_instance(service_name=service_name, ip=ip, port=port, weight=weight, healthy=True, enable=True)
    if not nacos_conn_status:
        ia_logging.info('nacos client connect failed')
    else:
        asyncio.run(nacos_heartbeat())
async def nacos_heartbeat():
    while True:
        client.send_heartbeat(service_name=service_name, ip=ip, port=port, weight=weight)
        await asyncio.sleep(10.0)
        ia_logging.info('nacos heartbeat')

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(nacos_client_connect)
except:
    ia_logging.info("nacos client failed to initialize")
