import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "uvicorn", "backend.main:app",
     "--reload", "--host", "0.0.0.0", "--port", "8000"],
    cwd=str(__import__("pathlib").Path(__file__).parent.parent),
)
