import argparse
import torch
import json
from typing import Dict, List
from dataclasses import dataclass, field, asdict
from lightning_fabric.accelerators import cuda


@dataclass
class Report:
    is_cuda_available: bool = False
    num_cuda_devices: int = 0

    available_devices: List[int] = field(default_factory=list)
    unavailable_devices: List[int] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    additional_context: Dict = field(default_factory=dict)


def main() -> None:
    import sys 

    print("ARGS", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")
    parser.add_argument(
        '--output-file', '--output_file', type=str, default=None, required=False, 
        help="Output file containing GPU check results. If none, results will be written to a console."
    )
    args = parser.parse_args()
    
    report = Report(is_cuda_available=cuda.is_cuda_available())
    if report.is_cuda_available:
        report.num_cuda_devices = cuda.num_cuda_devices()
        report.additional_context["CUDAAccelerator.is_available"] = (
            cuda.CUDAAccelerator.is_available()
        )

        visible_devices: list[int] = cuda._get_all_visible_cuda_devices()
        for gpu_idx in visible_devices:
            try:
                torch.tensor(0, device=torch.device("cuda", gpu_idx))
            except RuntimeError as err:
                report.error_messages.append(str(err))
                report.unavailable_devices.append(gpu_idx)
                continue

            report.available_devices.append(gpu_idx)

    if not args.output_file:
        print(json.dumps(asdict(report), indent=4, sort_keys=False))
    else:
        with open(args.output_file, "wt") as f:
            json.dump(asdict(report), f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()
