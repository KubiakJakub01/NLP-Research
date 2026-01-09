import io
import os
import subprocess
import tarfile
import tempfile

import docker

_docker_client = None

VALID_EXECUTION_BACKENDS = {
    'docker',
    'dangerously_use_uv',
    'dangerously_use_local_jupyter',
}

_default_backend = os.environ.get('PYTHON_EXECUTION_BACKEND', 'docker')
if _default_backend not in VALID_EXECUTION_BACKENDS:
    _default_backend = 'docker'

PYTHON_EXECUTION_BACKEND = _default_backend


def call_python_script(script: str) -> str:
    """
    Call a python script by writing it to a file in the container and executing it.
    """
    global _docker_client  # pylint: disable=global-statement
    if _docker_client is None:
        _docker_client = docker.from_env()
        # pull image `python:3.11` if not present
        try:
            _docker_client.images.get('python:3.11')
        except docker.errors.ImageNotFound:
            _docker_client.images.pull('python:3.11')

    # 1. Create a temporary tar archive containing the script
    script_name = 'script.py'
    tarstream = io.BytesIO()
    with tarfile.open(fileobj=tarstream, mode='w') as tar:
        script_bytes = script.encode('utf-8')
        tarinfo = tarfile.TarInfo(name=script_name)
        tarinfo.size = len(script_bytes)
        tar.addfile(tarinfo, io.BytesIO(script_bytes))
    tarstream.seek(0)

    # 2. Start the container
    container = _docker_client.containers.create(
        'python:3.11', command='sleep infinity', detach=True
    )
    try:
        container.start()
        # 3. Put the script into the container
        container.put_archive(path='/tmp', data=tarstream.read())
        # 4. Execute the script
        exec_result = container.exec_run(f'python /tmp/{script_name}')
        output = exec_result.output.decode('utf-8')
        if not output.strip():
            output = (
                '[WARN] No output available. '
                'Use print() to output anything to stdout to receive the output'
            )
    finally:
        container.remove(force=True)
    return output


def call_python_script_with_uv(script: str) -> str:
    """
    Call a python script by writing it to a file to a temporary directory
    and executing it with uv.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, 'script.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        exec_result = subprocess.run(
            ['uv', 'run', '--no-project', 'python', script_path],
            capture_output=True,
            check=True,
        )
        return (
            exec_result.stdout.decode('utf-8')
            if exec_result.returncode == 0
            else exec_result.stderr.decode('utf-8')
        )
