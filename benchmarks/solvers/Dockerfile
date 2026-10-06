# Solver comparison against Ceres Solver, on top of the RapidGLM image:
#   docker build -t rapidglm .
#   docker build -t rapidglm-solvers -f benchmarks/solvers/Dockerfile .
#   docker run --rm -v /path/to/dataset:/data rapidglm-solvers \
#       build-solvers/rapidglm_solver_benchmark /data fixed 1 > result.json
FROM rapidglm
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libceres-dev \
    && rm -rf /var/lib/apt/lists/*
RUN cmake -S . -B build-solvers -DCMAKE_BUILD_TYPE=Release -DRAPIDGLM_TESTS=OFF -DRAPIDGLM_SOLVER_BENCHMARK=ON \
    && cmake --build build-solvers --target rapidglm_solver_benchmark -j "$(nproc)"
