# mumax3 built from source against CUDA 13 (multi-arch: amd64 and arm64, e.g. DGX Spark / GB10).
#
#   docker build -t lemurpwned/mumax3:3.12-cuda13 benchmarks/mumax3
#   docker run --rm --gpus all -v $PWD:/work -w /work lemurpwned/mumax3:3.12-cuda13 mumax3 sp4.mx3
#
# CUDA 13 changed cuCtxCreate (extra CUctxCreateParams argument); the sed below passes nil.
ARG CUDA=13.0.0
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG GO=1.23.4
ARG MUMAX_REF=54efa12420b2c8d55fc08085b20b3822c8ad67d7
ARG CUDA_CC="75 80 86 89 90 100 120 121"
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://go.dev/dl/go${GO}.linux-$(dpkg --print-architecture).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:/usr/local/cuda/bin:$PATH GOPATH=/go CUDA_HOME=/usr/local/cuda \
    CGO_CFLAGS_ALLOW="(-fno-schedule-insns|-malign-double|-ffast-math)"
RUN git clone https://github.com/mumax/3.git /src && cd /src && git checkout ${MUMAX_REF} \
 && sed -i 's/C.cuCtxCreate(&ctx, C.uint(flags)/C.cuCtxCreate(\&ctx, nil, C.uint(flags)/' cuda/cu/context.go \
 && cd cuda && make realclean >/dev/null && make CUDA_CC="${CUDA_CC}" \
 && cd .. && go install github.com/mumax/3/cmd/mumax3 github.com/mumax/3/cmd/mumax3-convert

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
COPY --from=build /go/bin/ /usr/local/bin/
ENTRYPOINT []
CMD ["mumax3", "-v"]
