MAX_ACTIVE_TRIPS = [2, 3]
SAVINGS_THRESHOLD_B = [0.1, 0.2, 0.3, 0.5]


rule all:
    input:
        expand(
            "outputs/pool/mat{mat}_stb{stb}/run.csv",
            mat=MAX_ACTIVE_TRIPS,
            stb=SAVINGS_THRESHOLD_B,
        ),
        expand(
            "outputs/pool/mat{mat}_stb{stb}/ate.csv",
            mat=MAX_ACTIVE_TRIPS,
            stb=SAVINGS_THRESHOLD_B,
        ),
        "outputs/pool/summary.csv",
        "outputs/pool/summary.png",


rule run_estimators:
    output:
        "outputs/pool/mat{mat}_stb{stb}/run.csv",
    shell:
        """
        python scripts/run.py \
            --config-name=dq_expts \
            'env_params.env_params.max_active_trips={wildcards.mat}' \
            'env.savings_threshold_B={wildcards.stb}' \
            'run.output_path={output}' \
            'hydra.run.dir=.hydra/mat{wildcards.mat}_stb{wildcards.stb}'
        """


rule summarize:
    input:
        run=expand(
            "outputs/pool/mat{mat}_stb{stb}/run.csv",
            mat=MAX_ACTIVE_TRIPS,
            stb=SAVINGS_THRESHOLD_B,
        ),
        ate=expand(
            "outputs/pool/mat{mat}_stb{stb}/ate.csv",
            mat=MAX_ACTIVE_TRIPS,
            stb=SAVINGS_THRESHOLD_B,
        ),
    output:
        csv="outputs/pool/summary.csv",
        plot="outputs/pool/summary.png",
    shell:
        "python scripts/summarize.py {output.csv} {output.plot}"


rule compute_ate:
    output:
        "outputs/pool/mat{mat}_stb{stb}/ate.csv",
    shell:
        """
        python scripts/compute-ate.py \
            --config-name=dq_expts \
            'env_params.env_params.max_active_trips={wildcards.mat}' \
            'env.savings_threshold_B={wildcards.stb}' \
            'ate.output_path={output}' \
            'hydra.run.dir=.hydra/mat{wildcards.mat}_stb{wildcards.stb}'
        """
