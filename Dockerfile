# RapidGLM: library, CLI, C interface, Python and R layers, and their tests.
#   docker build -t rapidglm .
#   docker run --rm rapidglm                  # runs every test suite (scripts/container_tests.sh)
#   docker run --rm -v "$PWD":/data rapidglm rapidglm bulk-de /data/config.json
# Singularity/Apptainer: singularity build rapidglm.sif docker-daemon://rapidglm:latest
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ make libboost-dev liblapacke-dev libopenblas-pthread-dev \
        python3 python3-venv python3-numpy python3-pandas python3-scipy python3-patsy \
        r-base-core r-base-dev r-cran-cpp11 r-cran-matrix r-cran-mass \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Outer OpenMP loops call BLAS concurrently; one BLAS thread per caller (see README).
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /opt/rapidglm
COPY . .

# Library, CLI and C interface into /usr/local (librapidglm_c on the loader path,
# so the Python layer finds it without RAPIDGLM_C_LIBRARY).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j "$(nproc)" \
    && cmake --install build --prefix /usr/local \
    && ldconfig

# Python layer (pure Python over the C interface) and the R package.
ENV PYTHONPATH=/opt/rapidglm/bindings/python
RUN R CMD INSTALL bindings/r

# PyDESeq2 (MIT) in its own environment, for the RapidGLM inference back-end test.
ARG WITH_PYDESEQ2=1
RUN if [ "$WITH_PYDESEQ2" = 1 ]; then \
        python3 -m venv /opt/pydeseq2 && /opt/pydeseq2/bin/pip install --no-cache-dir --quiet pydeseq2; \
    fi

CMD ["scripts/container_tests.sh"]
