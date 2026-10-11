"""The MSD tag container of ``.sm`` / ``.ssc`` files.

Tags are tokenized by ``msdparser`` (escapes ``\\``, ``:``, ``;`` and ``//``
are handled there in both directions); this module only groups them into a
song tag table and per-chart tag tables, following StepMania's layout:

* ``.sm``: song tags, then one ``#NOTES:stepstype:description:difficulty:
  meter:radarvalues:notedata;`` parameter per chart.
* ``.ssc``: song tags, then for each chart a ``#NOTEDATA:;`` marker followed by
  the chart's own tags, ending with ``#NOTES`` (or ``#NOTES2``).

Keys are upper-cased; later duplicates win. A tag's value is its first
component, except for the multi-value tags (``ATTACKS``, ``DISPLAYBPM``),
whose components are joined with ``:`` and split again on write.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from msdparser import MSDParameter, MSDParserError, parse_msd

MULTI_VALUE = frozenset({"ATTACKS", "DISPLAYBPM"})
SM_CHART_FIELDS = ("STEPSTYPE", "DESCRIPTION", "DIFFICULTY", "METER", "RADARVALUES", "NOTES")


class Tags(dict[str, str]):
    """Ordered tag table of a song or chart."""

    extradata: list[str] | None = None
    """``.sm`` only: ``#NOTES`` components beyond the six standard fields."""


class Document:
    def __init__(self, ssc: bool) -> None:
        self.ssc = ssc
        self.tags = Tags()
        self.charts: list[Tags] = []


def _value(param: MSDParameter) -> str:
    key = param.key.upper()
    if key in MULTI_VALUE:
        return ":".join(param.components[1:])
    return param.value or ""


def parse(text: str, *, ssc: bool) -> Document:
    doc = Document(ssc)
    try:
        params: Iterator[MSDParameter] = parse_msd(string=text, ignore_stray_text=True)
        chart: Tags | None = None
        for param in params:
            key = param.key.upper()
            if ssc and key == "NOTEDATA":
                chart = Tags()
                doc.charts.append(chart)
            elif chart is not None:
                chart[key] = _value(param)
            elif not ssc and key == "NOTES":
                doc.charts.append(_sm_chart(param.components[1:]))
            else:
                doc.tags[key] = _value(param)
    except MSDParserError as exc:
        raise ValueError(str(exc)) from exc
    return doc


def _sm_chart(values: Sequence[str]) -> Tags:
    if len(values) < len(SM_CHART_FIELDS):
        raise ValueError(f"#NOTES needs {len(SM_CHART_FIELDS)} components, got {len(values)}")
    chart = Tags(zip(SM_CHART_FIELDS, (v.strip() for v in values), strict=False))
    if len(values) > len(SM_CHART_FIELDS):
        chart.extradata = list(values[len(SM_CHART_FIELDS) :])
    return chart


def _param(key: str, value: str) -> str:
    components = value.split(":") if key in MULTI_VALUE else [value]
    return str(MSDParameter((key, *components)))


def serialize(doc: Document) -> str:
    out = [_param(key, value) + "\n" for key, value in doc.tags.items()]
    out.append("\n")
    for chart in doc.charts:
        out.append(_serialize_ssc_chart(chart) if doc.ssc else _serialize_sm_chart(chart))
        out.append("\n")
    return "".join(out)


def _serialize_sm_chart(chart: Tags) -> str:
    fields = [f"\n     {chart.get(key, '')}" for key in SM_CHART_FIELDS[:-1]]
    fields.append(f"\n{chart.get('NOTES', '')}\n")
    return str(MSDParameter(("NOTES", *fields, *(chart.extradata or [])))) + "\n"


def _serialize_ssc_chart(chart: Tags) -> str:
    notes_key = next((k for k in ("NOTES", "NOTES2") if k in chart), "NOTES")
    out = [str(MSDParameter(("NOTEDATA", ""))) + "\n"]
    out += [_param(k, v) + "\n" for k, v in chart.items() if k != notes_key]
    out.append(_param(notes_key, chart.get(notes_key, "")) + "\n\n")
    return "".join(out)
