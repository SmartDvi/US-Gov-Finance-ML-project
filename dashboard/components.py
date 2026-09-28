"""Reusable building blocks shared by all dashboard pages."""

from __future__ import annotations

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd

# ── AG Grid column formatters (JavaScript run in the browser; d3 is built in) ──
PERCENT = {"function": "params.value == null ? '' : d3.format('.1%')(params.value)"}
PCT_POINTS = {"function": "params.value == null ? '' : d3.format('+.1f')(params.value * 100) + ' pp'"}
BILLIONS = {"function": "params.value == null ? '' : '$' + d3.format(',.2f')(params.value) + 'B'"}
DECIMAL_2 = {"function": "params.value == null ? '' : d3.format('.2f')(params.value)"}

# Green when above peers, red when below (used for "vs peers" columns)
GAP_STYLE = {
    "styleConditions": [
        {"condition": "params.value > 0.02", "style": {"color": "var(--mantine-color-teal-7)", "fontWeight": 600}},
        {"condition": "params.value < -0.02", "style": {"color": "var(--mantine-color-red-7)", "fontWeight": 600}},
    ]
}


# Colours given to chart series in order
SERIES_COLORS = ["indigo.6", "teal.6", "orange.6", "grape.6", "cyan.6", "red.6", "lime.7", "pink.6", "blue.8", "yellow.7"]


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts for the browser, with NaN turned into None (JSON null)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def page_header(title: str, description: str) -> dmc.Stack:
    return dmc.Stack(
        [dmc.Title(title, order=2), dmc.Text(description, c="dimmed", size="sm")],
        gap=4,
        mb="md",
    )


def kpi_card(label: str, value: str, note: str | None = None, color: str | None = None) -> dmc.Paper:
    return dmc.Paper(
        dmc.Stack(
            [
                dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600),
                # only pass a colour when one is given (None would reach the browser as null)
                dmc.Text(value, size="xl", fw=700, **({"c": color} if color else {})),
                dmc.Text(note, size="xs", c="dimmed") if note else None,
            ],
            gap=2,
        ),
        withBorder=True,
        p="md",
        radius="md",
    )


def kpi_row(cards: list) -> dmc.SimpleGrid:
    return dmc.SimpleGrid(cards, cols={"base": 1, "xs": 2, "md": len(cards)}, spacing="md", mb="md")


def section(title: str, children, description: str | None = None) -> dmc.Paper:
    header = [dmc.Title(title, order=4)]
    if description:
        header.append(dmc.Text(description, size="sm", c="dimmed"))
    return dmc.Paper(
        dmc.Stack([dmc.Stack(header, gap=2), children], gap="sm"),
        withBorder=True,
        p="md",
        radius="md",
        mb="md",
    )


def data_grid(grid_id: str, df: pd.DataFrame, column_defs: list[dict], height: int = 420) -> dag.AgGrid:
    """AG Grid with sorting, filtering, resizing and pagination switched on."""
    return dag.AgGrid(
        id=grid_id,
        rowData=to_records(df),
        columnDefs=column_defs,
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "minWidth": 110},
        columnSize="responsiveSizeToFit",
        # suppressFieldDotNotation: treat "tags.stage" as a column name, not a nested path
        dashGridOptions={"pagination": True, "paginationPageSize": 20, "animateRows": True,
                         "suppressFieldDotNotation": True},
        style={"height": height, "width": "100%"},
    )


def pipeline_missing() -> dmc.Alert:
    return dmc.Alert(
        "No pipeline outputs found. Run `uv run python main.py` first, then refresh this page.",
        title="Pipeline has not been run",
        color="yellow",
    )
