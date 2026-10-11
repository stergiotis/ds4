# HGN feasibility tools

Written from the clean-room [hgn-spec](https://github.com/jtsylve/hgn-spec)
(MIT or CC BY 4.0), which describes halogen's checkpoint format. Nothing
here comes from halogen's code.

- `hgn.py`: reads an HGN container's tensor table and payloads.
- `ht.py`: decodes a 2-D `ht` tensor (trellis codes, codebook 1, 128-point
  rotation) to dense values with numpy.
- `htmv.hip`: GPU matvec on `ht` payloads. `htmv w.bin N K su.f32 sv.f32
  x.f32 y.f32` checks one matrix (`HTV2=1` for the packed kernel);
  `htmv --bench N K copies [norot] [ct] [version]` measures bandwidth on a
  random payload.

Build: `hipcc -O3 --offload-arch=gfx1151 -o htmv htmv.hip`.
