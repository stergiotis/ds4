#!/bin/sh
# dskgen-contract.sh <remarks-file> <plan-header>...
#
# Checks a ROCm build against the register contract of dskgen plan headers:
# every `// dskgen-contract: <kernel> vgprs=<n> occupancy=<w>` line of the
# headers must be matched by the build's -Rpass-analysis=kernel-resource-usage
# remarks (hipcc's stderr, saved to <remarks-file>) for that kernel, with the
# same VGPR count and occupancy. Prints one line per kernel; exit 1 when any
# kernel changed or is missing, 2 on usage. Written by `dskgen emit-plan`.
[ $# -ge 2 ] || { echo "usage: $0 <remarks-file> <plan-header>..." >&2; exit 2; }
remarks=$1; shift
[ -r "$remarks" ] || { echo "$0: cannot read $remarks" >&2; exit 2; }
awk -v remarks="$remarks" '
/^\/\/ dskgen-contract: / {
    name = $3; v = $4; o = $5
    sub("vgprs=", "", v); sub("occupancy=", "", o)
    want_v[name] = v; want_o[name] = o; order[n++] = name
}
END {
    cur = ""
    while ((getline line < remarks) > 0) {
        if (index(line, "Function Name: ")) {
            cur = ""
            for (k in want_v) if (index(line, k "E")) cur = k
        } else if (cur != "" && index(line, " VGPRs: ")) {
            split(line, a, "VGPRs: "); split(a[2], b, " "); got_v[cur] = b[1]
        } else if (cur != "" && index(line, "Occupancy [waves/SIMD]: ")) {
            split(line, a, "SIMD]: "); split(a[2], b, " "); got_o[cur] = b[1]
        }
    }
    bad = 0
    for (i = 0; i < n; i++) {
        k = order[i]
        if (!(k in got_v) || !(k in got_o)) {
            printf("MISSING  %s: no kernel-resource-usage remark in %s\n", k, remarks); bad++
        } else if (got_v[k] != want_v[k] || got_o[k] != want_o[k]) {
            printf("CHANGED  %s: vgprs %s -> %s, occupancy %s -> %s\n", k, want_v[k], got_v[k], want_o[k], got_o[k]); bad++
        } else {
            printf("ok       %s: vgprs %s occupancy %s\n", k, got_v[k], got_o[k])
        }
    }
    exit bad ? 1 : 0
}' "$@"
