from pathlib import Path
import pandas as pd
import re
import numpy as np

def parse_xgemm_output(file_path: str) -> pd.DataFrame:
    """Parse the output of the xgemm testing script. The file contains the following section that needs
    to be converted to a pandas data frame:
    ```
    %   M     N     K   MAGMA Gflop/s (ms)  cuBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  cuBLAS error
    %========================================================================================================
     1088  1088  1088   3653.64  (   0.71)       65.16 (  39.53)     ---   (  ---  )    0.00e+00        ---    ok
     2112  2112  2112   12322.83 (   1.53)    18784.48 (   1.00)     ---   (  ---  )    0.00e+00        ---    ok
     3136  3136  3136   15663.43 (   3.94)    44567.25 (   1.38)     ---   (  ---  )    0.00e+00        ---    ok
     4160  4160  4160   16643.43 (   8.65)    52322.54 (   2.75)     ---   (  ---  )    0.00e+00        ---    ok
     5184  5184  5184   17481.95 (  15.94)    26293.14 (  10.60)     ---   (  ---  )    0.00e+00        ---    ok
     6208  6208  6208   17630.22 (  27.14)    60478.81 (   7.91)     ---   (  ---  )    0.00e+00        ---    ok
     7232  7232  7232   17060.41 (  44.34)    62868.32 (  12.03)     ---   (  ---  )    0.00e+00        ---    ok
     8256  8256  8256   17625.89 (  63.85)    62820.13 (  17.92)     ---   (  ---  )    0.00e+00        ---    ok
     9280  9280  9280   17696.02 (  90.32)    29389.58 (  54.39)     ---   (  ---  )    0.00e+00        ---    ok
     10304 10304 10304  16437.30 ( 133.11)    63110.88 (  34.67)     ---   (  ---  )    0.00e+00        ---    ok
    ```
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the start of the data section
    for i, line in enumerate(lines):
        if line.startswith("%======="):
            start_index = i + 1
            break
    else:
        raise ValueError("Data section not found in the file.")

    # Read the data into a DataFrame
    data: list[dict] = []
    for line in lines[start_index:]:
        line = line.strip()
        if not line:
            break
        parts: list[str] =  [part.strip() for part in re.split(r"[ \n\t()]+", line) if part.strip()]
        if len(parts) != 12:
            raise ValueError(f"Unexpected line format: line={line}, parts={parts}")
        
        data.append({
            "M": int(parts[0]),
            "N": int(parts[1]),
            "K": int(parts[2]),
            "magma_gflop_s": float(parts[3]),
            "magma_ms": float(parts[4]),
            "cublas_gflop_s": float(parts[5]),
            "cublas_ms": float(parts[6]),
            "magma_error": float(parts[9]),
            "cublas_error": parts[10],
            "status": parts[11].lower(),
            })

    return pd.DataFrame(data)
    

if __name__ == "__main__":
    benchmark_output_dir = Path("DIRECTORY_PATH_WITH_SGEMM_OR_DGEMM_RESULTS")
    csv_files: list[Path] = list(benchmark_output_dir.glob("*.csv"))

    results: dict[str, dict] = {}
    for csv_file in csv_files:
        df: pd.DataFrame = parse_xgemm_output(csv_file)
        parts: list[str] = csv_file.stem.split("_") # magma, dgemm, smc7, gpu, 2
        node, gpu_id = parts[2], parts[4]
        if node not in results:
            results[node] = {"node": node}
        results[node].update({gpu_id: float(df["cublas_ms"].mean())})
    
    df = pd.DataFrame(list(results.values()))
    df.sort_values(by=["node"], inplace=True)
    df = df[["node"] + sorted(df.columns[1:])]

    gpus = [str(i) for i in range(8)]
    
    df["mean"] = df[gpus].mean(axis=1)
    df["std_dev"] = df[gpus].std(axis=1)

    overall_mean: float = np.mean(df[gpus].values)
    overall_std: float = np.std(df[gpus].values)


    
    gpu_summaries = [
        dict(node="mean", **df[gpus].mean(axis=0).to_dict()),
        dict(node="std_dev", **df[gpus].std(axis=0).to_dict()),
    ]
    df = pd.concat([df, pd.DataFrame(gpu_summaries)], ignore_index=True)

    print(df)
    print("overall mean / std", overall_mean, overall_std)
