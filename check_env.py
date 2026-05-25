import shutil
import subprocess
import sys


def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def run_where(command):
    path = shutil.which(command)

    if path is None:
        print(f"{command}: NOT FOUND")
        return False

    print(f"{command}: {path}")
    return True


def run_version(command, args=None):
    if args is None:
        args = ["--version"]

    try:
        result = subprocess.run(
            [command] + args,
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip() or result.stderr.strip()

        if output:
            print(output)

        return result.returncode == 0
    except Exception as exc:
        print(f"Could not run {command}: {exc}")
        return False


def check_python():
    print_header("Python")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)


def check_torch():
    print_header("PyTorch / CUDA")

    try:
        import torch
    except Exception as exc:
        print("PyTorch: FAILED")
        print(exc)
        return

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def check_tools():
    print_header("Build tools")

    nvcc_found = run_where("nvcc")
    if nvcc_found:
        run_version("nvcc")

    print()

    cl_found = run_where("cl")
    if cl_found:
        run_version("cl", [])


def check_dcn():
    print_header("DCN / RFR imports")

    try:
        from model.dcn.modules.deform_conv import DeformConv
        print("DCN: OK")
    except Exception as exc:
        print("DCN: FAILED")
        print(exc)

    try:
        from model.RFR_framework import RFR
        print("RFR: OK")
    except Exception as exc:
        print("RFR: FAILED")
        print(exc)


def main():
    check_python()
    check_torch()
    check_tools()
    check_dcn()


if __name__ == "__main__":
    main()