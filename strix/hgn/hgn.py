"""Minimal HGN reader (from the hgn-spec container rules): hgn.table(path) -> {name: entry}"""
import struct, numpy as np
def table(path):
    f = open(path, 'rb'); h = f.read(0x68)
    magic, ver, n, toff, doff, fsize = struct.unpack('<IIQQQQ', h[:0x28])
    assert magic == 0x314E4748, hex(magic)
    f.seek(toff); out = {}
    for i in range(n):
        e = f.read(0xA0)
        name = e[:96].split(b'\0', 1)[0].decode()
        store, rank = struct.unpack('<II', e[0x60:0x68])
        shape = struct.unpack('<4q', e[0x68:0x88])[:rank]
        off, size, cks, var = struct.unpack('<QQII', e[0x88:0xA0])
        out[name] = dict(store=store, shape=shape, off=off, size=size, variant=var)
    return out, h[0x28:0x68].split(b'\0', 1)[0].decode()
def payload(path, e):
    return np.memmap(path, np.uint8, 'r', offset=e['off'], shape=(e['size'],))
