from pathlib import Path
from typing import Optional, Union, Text

import paramiko
from paramiko.client import SSHClient
from paramiko import BadHostKeyException, AuthenticationException, SSHException

from utils.decorator import inspect_param
from giturlparse import parse
from loguru import logger


class ServerTrainer:
    def __init__(
            self,
            host: str,
            port: Optional[int] = 22,
            ssh_username: Optional[str] = None,
            ssh_password: Optional[str] = None,
            timeout: Optional[int] = 30
    ):
        assert host is not None, "You must provide `host` to remote server"
        self.host = host
        self.port = port
        self.username = ssh_username
        self.password = ssh_password
        self.timeout = timeout
        self.ssh = self._login()

    def _login(self, **kwargs):
        """
        Connect to the SSH server and authenticate to it
        :param kwargs: including host, port, username, password of user logging to server
        :return:
        """
        host = kwargs["host"] if "host" in kwargs else self.host
        port = kwargs["port"] if "port" in kwargs else self.port
        username = kwargs["username"] if "username" in kwargs else self.username
        password = kwargs["password"] if "password" in kwargs else self.password

        try:
            ssh = SSHClient()
            # Auto add host to known hosts
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to server
            ssh.connect(host, port=port, username=username, password=password, timeout=self.timeout)
            logger.info(f"Remote to server {host} successfully")
            return ssh
        except(
                BadHostKeyException,
                SSHException,
                AuthenticationException
        ) as e:
            raise ConnectionError(
                f"SSH Connection failed", e
            )

    def _logout(self):
        logger.info(f"The connection to server {self.host} will be closed")
        self.ssh.close()

    def _resource(self):
        # Get available memory
        cpu_stdout = self.execute_commands("free -h")
        cpu = []
        for line in cpu_stdout.readlines():
            cpu.append((self.remove_none_in_list(line)))
        available_cpu_memory = cpu[1][5].replace('Gi', '')

        # Get available VRam
        gpu = []
        gpu_stdout = self.execute_commands("nvidia-smi --query-gpu=memory.free --format=csv")
        for line in gpu_stdout.readlines():
            gpu.append(self.remove_none_in_list(line))
        available_gpu_memory = round(float(gpu[1][0]) / 1024, 2)

        return {
            "cpu": available_cpu_memory,
            "gpu": available_gpu_memory
        }

    # @inspect_param
    def clone_repository(
            self,
            git_username: str,
            git_password: str,
            repo: Union[str, Path],
            branch: Optional[str] = None
    ):  # type: ignore
        """
        Clone repository from git to user workspace on server
        This will remove the similar repository in server to avoid conflicting or mode pull code from git if you sure that
        any conflict is found.
        
        :param git_username: username
        :param git_password: password
        :param repo: repo that you want to pull
        :param branch: branch that you want to check out to train model
        # :param action: if action is pull, existing repository will not be removed instead of pull code from git
        :return:
        """
        info = parse(repo)
        new_https = f"https://{git_username}:{git_password}@{info.domain}{info.pathname}"

        # cmd = f"rm -rf {info.pathname} | git clone {new_https}"
        # self.execute_commands(cmd)
        #
        # logger.info(f"Repository `https://{info.domain}{info.pathname}` has been pulled")
        cmd = f"cd {info.repo} \n git branch -r"
        branches = self.execute_commands(cmd)
        for line in branches.readlines():
            print(line.rstrip(' ').split('\n'))

    def execute_commands(self, command: Union[str, Text]):
        return self.ssh.exec_command(command)[1]

    @staticmethod
    def remove_none_in_list(line: str) -> list:
        return list(filter(None, (line.rstrip().split(' '))))


if __name__ == "__main__":
    h = "183.81.34.200"
    p = 2234
    user = "phongnt"
    pas = "Ftech@123"
    pg = "Since0804@@"
    server = ServerTrainer(host=h, port=p, ssh_username=user, ssh_password=pas)
    # print(server._resource(host=h, port=p, username=user, password=pas))
    server.clone_repository(user, pg, "https://gitlab.ftech.ai/nlp/va/knowledge-retrieval.git")
