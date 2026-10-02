"""Minimal reader of a profiler ``*.xplane.pb`` (tsl XSpace) in pure Python -- no protobuf
package here and jax 0.6.2's ``ProfileData`` is absent; the Chrome-trace export caps the
event count, so the device side of a long profile is only in the xplane.

Wire format of the fields read (tsl/profiler/protobuf/xplane.proto):
  XSpace:  1 planes (XPlane)
  XPlane:  1 id, 2 name, 3 lines (XLine), 4 event_metadata (map int64 -> XEventMetadata),
           5 stat_metadata (map int64 -> XStatMetadata)
  XLine:   1 id, 2 name, 3 timestamp_ns, 4 events (XEvent), 11 display_name
  XEvent:  1 metadata_id, 2 offset_ps, 3 duration_ps, 4 stats (XStat)
  XEventMetadata: 1 id, 2 name, 4 display_name, 5 stats
  XStatMetadata:  1 id, 2 name
  XStat:   1 metadata_id, 2 double, 3 uint64, 4 int64, 5 str, 6 bytes, 7 ref_value
"""

from __future__ import annotations

import struct


def _varint(buf, i):
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7


def fields(buf, start=0, end=None):
    """``(field, wire_type, value)`` of a message: ``value`` an int (varint / fixed) or a
    ``(start, end)`` span into ``buf`` (length-delimited)."""
    i = start
    end = len(buf) if end is None else end
    while i < end:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            value = (i, i + n)
            i += n
        elif wire == 1:
            value = struct.unpack_from("<d", buf, i)[0]
            i += 8
        elif wire == 5:
            value = struct.unpack_from("<f", buf, i)[0]
            i += 4
        else:
            raise ValueError(f"wire type {wire}")
        yield field, wire, value


def _text(buf, span):
    return bytes(buf[span[0] : span[1]]).decode("utf-8", "replace")


def _stat(buf, span, stat_names):
    """``(name, value)`` of an XStat (a ref_value resolved to its stat-metadata name)."""
    name, value = None, None
    for f, w, v in fields(buf, *span):
        if f == 1:
            name = stat_names.get(v, str(v))
        elif f == 7:
            value = ("ref", v)
        elif f == 5:
            value = _text(buf, v)
        elif f in (2, 3, 4):
            value = v
    if isinstance(value, tuple) and value[0] == "ref":
        value = stat_names.get(value[1], str(value[1]))
    return name, value


def planes(path):
    """Plane spans and names of the XSpace at ``path``: ``(buf, [(name, span)])``."""
    buf = memoryview(open(path, "rb").read())
    out = []
    for f, _w, span in fields(buf):
        if f != 1:
            continue
        name = ""
        for g, _w2, v in fields(buf, *span):
            if g == 2:
                name = _text(buf, v)
                break
        out.append((name, span))
    return buf, out


def plane_events(buf, span, want_line=None, event_stats=False):
    """Events of one plane: yields ``(line_name, event_name, start_ns, duration_ns, stats)``
    (``stats`` a dict of the event's own stats plus its metadata's, only with
    ``event_stats``); ``want_line(name) -> bool`` skips other lines unparsed."""
    meta_names, meta_stats, stat_names, lines = {}, {}, {}, []
    for f, _w, v in fields(buf, *span):
        if f == 5:  # stat metadata map entry
            key, sm = None, None
            for g, _w2, x in fields(buf, *v):
                if g == 1:
                    key = x
                elif g == 2:
                    sm = x
            for g, _w2, x in fields(buf, *sm):
                if g == 2:
                    stat_names[key] = _text(buf, x)
        elif f == 4:
            key, em = None, None
            for g, _w2, x in fields(buf, *v):
                if g == 1:
                    key = x
                elif g == 2:
                    em = x
            meta_names[key] = ""
            meta_stats[key] = []
            for g, _w2, x in fields(buf, *em):
                if g == 2:
                    meta_names[key] = _text(buf, x)
                elif g == 5:
                    meta_stats[key].append(x)
        elif f == 3:
            lines.append(v)
    resolved = {}
    for lspan in lines:
        lname, t0, events = "", 0, []
        for g, _w, x in fields(buf, *lspan):
            if g == 2:
                lname = _text(buf, x)
            elif g == 3:
                t0 = x
            elif g == 4:
                events.append(x)
        if want_line is not None and not want_line(lname):
            continue
        for espan in events:
            mid = off = dur = 0
            own = []
            for g, _w, x in fields(buf, *espan):
                if g == 1:
                    mid = x
                elif g == 2:
                    off = x
                elif g == 3:
                    dur = x
                elif g == 4 and event_stats:
                    own.append(x)
            stats = None
            if event_stats:
                if mid not in resolved:
                    resolved[mid] = dict(_stat(buf, s, stat_names) for s in meta_stats.get(mid, []))
                stats = dict(resolved[mid])
                stats.update(_stat(buf, s, stat_names) for s in own)
            yield lname, meta_names.get(mid, str(mid)), t0 + off / 1e3, dur / 1e3, stats
