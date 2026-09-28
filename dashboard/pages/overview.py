"""Page 1 — Overview: headline numbers and a state-by-state trend explorer."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback

from dashboard import data
from dashboard.components import (
    BILLIONS, PERCENT, SERIES_COLORS, data_grid, kpi_card, kpi_row, page_header,
    pipeline_missing, section, to_records,
)

dash.register_page(
    __name__, path="/", name="Overview", description="Headline numbers & state explorer", order=0
)

SNAPSHOT_COLUMNS = [
    {"field": "state", "headerName": "State", "pinned": "left", "minWidth": 150},
    {"field": "general_revenue_bn", "headerName": "General revenue", "valueFormatter": BILLIONS},
    {"field": "totals_general_revenue_growth", "headerName": "Revenue growth", "valueFormatter": PERCENT},
    {"field": "expenditure_to_revenue_ratio", "headerName": "Spending / revenue", "valueFormatter": PERCENT},
    {"field": "debt_to_revenue_ratio", "headerName": "Debt / revenue", "valueFormatter": PERCENT},
    {"field": "welfare_share", "headerName": "Welfare share", "valueFormatter": PERCENT},
    {"field": "education_share", "headerName": "Education share", "valueFormatter": PERCENT},
    {"field": "intergovernmental_dependency", "headerName": "Federal aid share", "valueFormatter": PERCENT},
]


def layout(**_):
    if not data.pipeline_has_run():
        return pipeline_missing()

    df = data.features()
    forecast = data.revenue_forecast()
    anomalies = data.pipeline_metrics()["anomaly_detection"]
    latest_year = int(df["year"].max())
    latest = df[df["year"] == latest_year]

    revenue_now = latest["totals_general_revenue"].sum() * data.THOUSANDS_TO_BILLIONS
    revenue_next = forecast["totals_general_revenue_forecast"].sum() * data.THOUSANDS_TO_BILLIONS
    biggest_states = latest.nlargest(3, "totals_general_revenue")["state"].tolist()

    return dmc.Container(
        [
            page_header(
                "Overview",
                "US state government finances, 1992–2019 (US Census, thousands of USD converted to $ billions).",
            ),
            kpi_row([
                kpi_card("States", str(df["state"].nunique())),
                kpi_card("Years covered", f"{df['year'].min()}–{latest_year}",
                         f"{df['year'].nunique()} years · 2005–2011 not in source"),
                kpi_card(f"General revenue {latest_year}", f"${revenue_now:,.0f}B", "all states combined"),
                kpi_card(f"Forecast {latest_year + 1}", f"${revenue_next:,.0f}B",
                         f"{revenue_next / revenue_now - 1:+.1%} vs {latest_year}", color="indigo"),
                kpi_card("Anomalies flagged", str(anomalies["anomalies_flagged"]),
                         f"{anomalies['anomaly_rate_pct']}% of state-years", color="red"),
            ]),
            section(
                "State explorer",
                dmc.Stack([
                    dmc.Group(
                        [
                            dmc.Select(
                                id="overview-metric",
                                label="Metric",
                                data=[{"value": col, "label": label} for label, col in data.MONEY_METRICS.items()],
                                value="totals_general_revenue",
                                allowDeselect=False,
                                w=240,
                            ),
                            dmc.MultiSelect(
                                id="overview-states",
                                label="States",
                                data=data.state_names(),
                                value=biggest_states,
                                searchable=True,
                                clearable=True,
                                maxValues=len(SERIES_COLORS),
                                placeholder="Pick up to 10 states",
                                style={"flex": 1, "minWidth": 260},
                            ),
                        ],
                        align="flex-end",
                    ),
                    dmc.LineChart(
                        id="overview-trend",
                        h=340,
                        dataKey="year",
                        data=[],
                        series=[],
                        curveType="monotone",
                        withLegend=True,
                        connectNulls=False,
                        valueFormatter={"function": "billions"},
                    ),
                ]),
                description="Compare states over time. Values in $ billions (nominal).",
            ),
            section(
                f"Snapshot of selected states — {latest_year}",
                data_grid("overview-grid", latest.iloc[0:0], SNAPSHOT_COLUMNS, height=300),
            ),
        ],
        fluid=True,
    )


@callback(
    Output("overview-trend", "data"),
    Output("overview-trend", "series"),
    Output("overview-grid", "rowData"),
    Input("overview-metric", "value"),
    Input("overview-states", "value"),
)
def update_explorer(metric, states):
    states = states or []
    df = data.features()
    selected = df[df["state"].isin(states)]

    # Wide table for the chart: one row per year, one column per state
    wide = (
        selected.assign(value=selected[metric] * data.THOUSANDS_TO_BILLIONS)
        .pivot(index="year", columns="state", values="value")
        .round(2)
        .reset_index()
    )
    series = [{"name": s, "color": SERIES_COLORS[i % len(SERIES_COLORS)]} for i, s in enumerate(states)]

    latest = selected[selected["year"] == df["year"].max()].copy()
    latest["general_revenue_bn"] = latest["totals_general_revenue"] * data.THOUSANDS_TO_BILLIONS
    return to_records(wide), series, to_records(latest)
