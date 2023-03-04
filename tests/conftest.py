import pytest
from IPython.testing.globalipapp import start_ipython


@pytest.fixture(scope="session")
def session_ip():
    yield start_ipython()


@pytest.fixture(scope="function")
def ip(session_ip):
    session_ip.run_line_magic(magic_name="load_ext", line="genai")
    yield session_ip
    session_ip.run_line_magic(magic_name="unload_ext", line="genai")
    session_ip.run_line_magic(magic_name="reset", line="-f")

    # Reset the history manager so that the next test starts with a clean slate
    session_ip.history_manager.reset()
