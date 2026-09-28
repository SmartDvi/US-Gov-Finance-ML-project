"""Page 2 — Revenue forecast: next-year general revenue per state, and model accuracy."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, callback

from dashboard import data
from dashboard.components import (
    BILLIONS, PERCENT, data_grid, kpi_card, kpi_row, page_header, pipeline_missing,
    section, to_records,
)

dash.register_page(
    __name__, path="/forecast", name="Revenue Forecast",
    description="Next-year revenue per state", order=1,
)

VIEWS = {
    "all": "All selected states",
    "weakest": "10 weakest growth",
    "strongest": "10 strongest growth",
}

GRID_COLUMNS = [
    {"field": "state", "headerName": "State", "pinned": "left", "minWidth": 150},
    {"field": "base_revenue_bn", "headerName": "Revenue (base year)", "valueFormatter": BILLIONS},
    {"field": "predicted_growth", "headerName": "Predicted growth", "valueFormatter": PERCENT,
     "cellStyle": {"styleConditions": [
         {"condition": "params.value < 0.03", "style": {"color": "var(--mantine-color-red-7)", "fontWeight": 600}}
     ]}},
    {"field": "forecast_revenue_bn", "headerName": "Forecast revenue", "valueFormatter": BILLIONS},
    {"field": "forecast_year", "headerName": "Forecast year"},
]


def _forecast_table() -> pd.DataFrame:
    df = data.revenue_forecast()
    df["base_revenue_bn"] = df["totals_general_revenue_base_year"] * data.THOUSANDS_TO_BILLIONS
    df["forecast_revenue_bn"] = df["totals_general_revenue_forecast"] * data.THOUSANDS_TO_BILLIONS
    return df.sort_values("predicted_growth")


def layout(**_):
    if not data.pipeline_has_run():
        return pipeline_missing()

    test = data.pipeline_metrics()["forecaster"]["test"]
    model, naive, average = (test[k] for k in ("random_forest", "baseline_naive_no_change", "baseline_average_growth"))
    states = data.state_names()

    return dmc.Container(
        [
            page_header(
                "Revenue Forecast",
                "Random forest (Spark ML) predicting next year's general-revenue growth per state. "
                "Accuracy measured on 2017–2019, years the model never saw during training.",
            ),
            kpi_row([
                kpi_card("Model error (MAPE)", f"{model['revenue_mape_pct']}%", "random forest, test years", color="indigo"),
                kpi_card("Baseline: no change", f"{naive['revenue_mape_pct']}%", "revenue stays flat"),
                kpi_card("Baseline: average growth", f"{average['revenue_mape_pct']}%", "historical average rate"),
                kpi_card("Champion model", f"v{data.champion_version() or '–'}", "MLflow model registry"),
            ]),
            section(
                "Forecast by state",
                dmc.Stack([
                    dmc.Group(
                        [
                            dmc.MultiSelect(
                                id="forecast-states",
                                label="States (empty = all)",
                                data=states,
                                searchable=True,
                                clearable=True,
                                placeholder="All states",
                                style={"flex": 1, "minWidth": 260},
                            ),
                            dmc.Select(
                                id="forecast-view",
                                label="Show",
                                data=[{"value": k, "label": v} for k, v in VIEWS.items()],
                                value="weakest",
                                allowDeselect=False,
                                w=220,
                            ),
                        ],
                        align="flex-end",
                    ),
                    dmc.BarChart(
                        id="forecast-bar",
                        h=320,
                        dataKey="state",
                        data=[],
                        series=[{"name": "predicted_growth", "label": "Predicted growth", "color": "indigo.6"}],
                        valueFormatter={"function": "percent"},
                        getBarColor={"function": "signColor"},
                        xAxisProps={"angle": -35, "textAnchor": "end", "height": 90, "interval": 0},
                    ),
                    data_grid("forecast-grid", _forecast_table(), GRID_COLUMNS, height=360),
                ]),
            ),
            section(
                "History and forecast for one state",
                dmc.Stack([
                    dmc.Select(
                        id="forecast-state",
                        label="State",
                        data=states,
                        value="CALIFORNIA" if "CALIFORNIA" in states else states[0],
                        searchable=True,
                        allowDeselect=False,
                        w=260,
                    ),
                    dmc.LineChart(
                        id="forecast-history",
                        h=320,
                        dataKey="year",
                        data=[],
                        series=[
                            {"name": "actual", "label": "Actual", "color": "gray.7"},
                            {"name": "forecast", "label": "Model forecast", "color": "indigo.6"},
                        ],
                        connectNulls=True,
                        withLegend=True,
                        valueFormatter={"function": "billions"},
                    ),
                ]),
                description="Grey = reported revenue. Blue = forecasts for the test years (2017–2019) "
                            "and for the year after the data ends.",
            ),
        ],
        fluid=True,
    )


@callback(
    Output("forecast-bar", "data"),
    Output("forecast-grid", "rowData"),
    Input("forecast-states", "value"),
    Input("forecast-view", "value"),
)
def update_forecast_table(states, view):
    df = _forecast_table()
    if states:
        df = df[df["state"].isin(states)]
    if view == "weakest":
        df = df.nsmallest(10, "predicted_growth")
    elif view == "strongest":
        df = df.nlargest(10, "predicted_growth").sort_values("predicted_growth")
    records = to_records(df)
    return records, records


@callback(Output("forecast-history", "data"), Input("forecast-state", "value"))
def update_state_history(state):
    history = data.features()
    history = history[history["state"] == state][["year", "totals_general_revenue"]]
    actual = history.rename(columns={"totals_general_revenue": "actual"})

    test = data.forecaster_test_predictions()
    test = test[test["state"] == state][["target_year", "forecast_value"]]
    future = data.revenue_forecast()
    future = future[future["state"] == state][["forecast_year", "totals_general_revenue_forecast"]]
    forecasts = pd.concat([
        test.set_axis(["year", "forecast"], axis=1),
        future.set_axis(["year", "forecast"], axis=1),
    ])

    chart = actual.merge(forecasts, on="year", how="outer").sort_values("year")
    chart[["actual", "forecast"]] = (chart[["actual", "forecast"]] * data.THOUSANDS_TO_BILLIONS).round(2)
    return to_records(chart)
