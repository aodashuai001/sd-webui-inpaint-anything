from modules import devices, script_callbacks, shared
from fastapi import FastAPI, Body
import gradio as gr
import numpy as np
from nacos_client.client import NacosClient
import threading
import time
from ia_logging import ia_logging
import socket
import sys
from ia_config import IAConfig, get_ia_config
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('114.114.114.114', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

server_address = 'http://120.78.215.48:8848'
namespace = '3bcb0b1d-1c2b-4069-97df-9612dac5005c'
username = ''  # 可选，访问密钥
password = ''  # 可选，访问密钥
nacos_enable = False

service_name = 'sd-service'  # 注册的服务名
# ip = get_local_ip()  # 本机IP地址
# port = 7861  # 服务端口号
ip = "https://4517q65q57.zicp.fun"
port = 0
weight = 1.0  # 权重，默认为1.0
# cluster_name = 'dev'  # 可选，集群名，默认为'default'
metadata = {}  # 可选，元数据，用于自定义信息
def set_nacos_config():
    global server_address, namespace, username, password, ip, port, nacos_enable
    profiles = 'DEFAULT'
    # ia_logging.info(sys.argv)
    for idx in range(len(sys.argv)):
        arg = sys.argv[idx]
        if arg.startswith('--profiles'):
            profiles = sys.argv[idx + 1].upper()
            break
    ip = get_ia_config('nacos_client_ip', profiles)
    if ip == 'local':
        ip = get_local_ip()
        port = 7861
    else:
        port = get_ia_config('nacos_client_port', profiles)
        
    ia_logging.info(f'current profiles is {profiles}')
    server_address = get_ia_config('nacos_server', profiles)
    namespace = get_ia_config('nacos_namespace', profiles)
    username = get_ia_config('nacos_username', profiles)
    password = get_ia_config('nacos_password', profiles)
    password = get_ia_config('nacos_password', profiles)
    nacos_enable = get_ia_config('nacos_enable', profiles)
def nacos_client_connect(_: gr.Blocks, app: FastAPI):
    global client
    set_nacos_config()
    if not nacos_enable:
        return
    ia_logging.info(f'nacos client start connecting:{server_address},{namespace}')
    client = NacosClient(server_addresses=server_address, namespace=namespace, password=f"{password}", username=f"{username}")
    nacos_conn_status = client.add_naming_instance(service_name=service_name, ip=ip, port=port, weight=weight, healthy=True, enable=True)
    if not nacos_conn_status:
        ia_logging.info('nacos client connect failed')
    else:
        nacos_thread = threading.Thread(target=nacos_heartbeat)
        # 启动线程
        nacos_thread.start()
def nacos_heartbeat():
    while True:
        try:
            client.send_heartbeat(service_name=service_name, ip=ip, port=port, weight=weight)
            time.sleep(10.0)  # 线程休眠2秒
            ia_logging.info('nacos heartbeat')
        except Exception:
            ia_logging.info('nacos client send heartbeat error')


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(nacos_client_connect)
except:
    ia_logging.info("nacos client failed to initialize")
