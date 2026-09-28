"""Page 4 — Anomaly audit: unusual year-over-year swings, with the reason for each flag."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, callback

from dashboard import data
from dashboard.components import (
    DECIMAL_2, PERCENT, data_grid, kpi_card, kpi_row, page_header, pipeline_missing,
    section, to_records,
)

dash.register_page(
    __name__, path="/anomalies", name="Anomaly Audit",
    description="Unusual swings to review", order=3,
)

FEATURES = data.CONFIG["anomaly"]["features"]
THRESHOLD = data.CONFIG["anomaly"]["z_threshold"]

# Short readable labels, e.g. "details_interest_on_general_debt_growth" -> "Interest on general debt growth"
LABELS = {f: data.pretty(f.removeprefix("totals_").removeprefix("details_")) for f in FEATURES}

GRID_COLUMNS = [
    {"field": "state", "headerName": "State", "pinned": "left", "minWidth": 150},
    {"field": "year", "headerName": "Year", "maxWidth": 100},
    {"field": "anomaly_score", "headerName": "Score (max |z|)", "valueFormatter": DECIMAL_2, "sort": "desc"},
    {"field": "reasons", "headerName": "Why it was flagged (z-score)", "minWidth": 380,
     "wrapText": True, "autoHeight": True},
] + [
    {"field": f, "headerName": LABELS[f], "valueFormatter": PERCENT, "minWidth": 140}
    for f in FEATURES
]


def _filtered(states, metrics, years, view) -> pd.DataFrame:
    df = data.anomaly_scores()
    df = df[df["year"].between(years[0], years[1])]
    if view == "flagged":
        df = df[df["is_anomaly"] == 1]
    if states:
        df = df[df["state"].isin(states)]
    if metrics:  # keep rows where at least one selected metric is beyond the threshold
        df = df[(df[[f"z_{m}" for m in metrics]].abs() > THRESHOLD).any(axis=1)]
    for feature, label in LABELS.items():
        df["reasons"] = df["reasons"].str.replace(feature, label, regex=False)
    return df.sort_values("anomaly_score", ascending=False)


def layout(**_):
    if not data.pipeline_has_run():
        return pipeline_missing()

    scores = data.anomaly_scores()
    min_year, max_year = int(scores["year"].min()), int(scores["year"].max())

    return dmc.Container(
        [
            page_header(
                "Anomaly Audit",
                f"Robust z-score on year-over-year changes: z = 0.6745 × (value − median) / MAD. "
                f"A state-year is flagged when any |z| > {THRESHOLD}. Unusual ≠ fraud; each flag lists its reason.",
            ),
            section(
                "Filters",
                dmc.Stack([
                    dmc.SimpleGrid(
                        [
                            dmc.MultiSelect(id="anomaly-states", label="States", data=data.state_names(),
                                            searchable=True, clearable=True, placeholder="All states"),
                            dmc.MultiSelect(
                                id="anomaly-metrics", label="Triggered by metric",
                                data=[{"value": f, "label": LABELS[f]} for f in FEATURES],
                                clearable=True, placeholder="Any metric",
                            ),
                            dmc.Select(
                                id="anomaly-view", label="Rows",
                                data=[{"value": "flagged", "label": "Flagged only"},
                                      {"value": "all", "label": "All scored state-years"}],
                                value="flagged", allowDeselect=False,
                            ),
                        ],
                        cols={"base": 1, "md": 3},
                    ),
                    dmc.Text("Years", size="sm", fw=500),
                    dmc.RangeSlider(
                        id="anomaly-years", min=min_year, max=max_year, value=[min_year, max_year],
                        minRange=0, marks=[{"value": y, "label": str(y)} for y in (min_year, 2004, 2012, max_year)],
                        mb="lg",
                    ),
                ]),
            ),
            dmc.Box(id="anomaly-kpis"),
            section(
                "Flagged state-years per year",
                dmc.BarChart(
                    id="anomaly-by-year", h=260, dataKey="year", data=[],
                    series=[{"name": "flagged", "label": "Flagged", "color": "red.6"}],
                ),
            ),
            section("Audit list", data_grid("anomaly-grid", scores.iloc[0:0], GRID_COLUMNS, height=480)),
        ],
        fluid=True,
    )


@callback(
    Output("anomaly-grid", "rowData"),
    Output("anomaly-by-year", "data"),
    Output("anomaly-kpis", "children"),
    Input("anomaly-states", "value"),
    Input("anomaly-metrics", "value"),
    Input("anomaly-years", "value"),
    Input("anomaly-view", "value"),
)
def update_audit(states, metrics, years, view):
    df = _filtered(states, metrics, years, view)
    flagged = df[df["is_anomaly"] == 1]
    per_year = flagged.groupby("year").size().rename("flagged").reset_index()

    kpis = kpi_row([
        kpi_card("Rows shown", f"{len(df):,}"),
        kpi_card("Flagged", f"{len(flagged):,}", color="red"),
        kpi_card("States affected", str(flagged["state"].nunique())),
        kpi_card("Highest score", f"{flagged['anomaly_score'].max():.1f}" if len(flagged) else "–"),
    ])
    return to_records(df), to_records(per_year), kpis
