import argparse
import subprocess


def main() -> None:
    import sys
    # hellow

    print("ARGS", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("mlcube_task", type=str, help="Task for this MLCube.")
    parser.add_argument(
        "--output-file",
        "--output_file",
        type=str,
        default=None,
        required=False,
        help="Output file containing GPU check results. If none, results will be written to a console.",
    )
    args = parser.parse_args()

    command = "nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,pstate,memory.total --format=csv"
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True)

        if args.output_file:
            with open(args.output_file, "wt", encoding="utf-8") as f:
                f.write(output)
            print(f"GPU information saved to {args.output_file}")
        else:
            print(output)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi command: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
