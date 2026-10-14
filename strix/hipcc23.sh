#!/bin/bash
# hipcc stand-in: AMD clang 23 (from the ROCm 10 toolbox image) with the host's
# ROCm 7.1 HIP headers and runtime.  Build:  make strix-halo HIPCC=strix/hipcc23.sh
# Toolchain (once, ~1 GB), into $DS4_LLVM23 (default ~/.local/share/ds4/rocm10-llvm):
#   cid=$(podman create docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0-strix-llama true)
#   podman cp $cid:/opt/rocm/core-10.0/lib/llvm $DS4_LLVM23/
#   podman cp $cid:/opt/rocm/core-10.0/lib/rocm_sysdeps $DS4_LLVM23/; podman rm $cid
#   mkdir -p $DS4_LLVM23/lib && cp -a $DS4_LLVM23/rocm_sysdeps/lib/librocm_sysdeps_z{,std}.so* $DS4_LLVM23/lib/
#   mkdir -p $DS4_LLVM23/linkroot/lib && ln -sf /usr/lib/x86_64-linux-gnu/libamdhip64.so $DS4_LLVM23/linkroot/lib/
D=${DS4_LLVM23:-$HOME/.local/share/ds4/rocm10-llvm}
export LD_LIBRARY_PATH=$D/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
args=(); link=1
for a in "$@"; do
  case "$a" in
    *.cu|*.hip) args+=(-x hip "$a");;
    -c) link=0; args+=("$a");;
    *) args+=("$a");;
  esac
done
extra=(--rocm-path=/usr --rocm-device-lib-path=$D/llvm/amdgcn/bitcode -mcode-object-version=5)
[ $link = 1 ] && extra=(--rocm-path=$D/linkroot --rocm-device-lib-path=$D/llvm/amdgcn/bitcode --hip-link -L/usr/lib/x86_64-linux-gnu)
exec $D/llvm/bin/clang++ "${extra[@]}" "${args[@]}"
