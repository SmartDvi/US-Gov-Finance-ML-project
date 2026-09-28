"""Page 3 — Peer clusters: which states are alike, and where a state differs from its peers."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback

from dashboard import data
from dashboard.components import (
    GAP_STYLE, PCT_POINTS, PERCENT, data_grid, page_header, pipeline_missing, section,
    to_records,
)

dash.register_page(
    __name__, path="/clusters", name="Peer Clusters",
    description="KMeans peer groups & benchmarks", order=2,
)

FEATURES = data.CONFIG["clustering"]["features"]
DEFAULT_FEATURES = ["welfare_share", "education_share", "health_share", "highways_share", "debt_to_revenue_ratio"]


def _column_defs(features: list[str]) -> list[dict]:
    cols = [
        {"field": "state", "headerName": "State", "pinned": "left", "minWidth": 150},
        {"field": "cluster_id", "headerName": "Cluster", "maxWidth": 110},
    ]
    for f in features:
        cols.append({"field": f, "headerName": data.pretty(f), "valueFormatter": PERCENT})
        cols.append({"field": f"{f}_vs_peers", "headerName": "vs peers", "valueFormatter": PCT_POINTS,
                     "cellStyle": GAP_STYLE, "headerTooltip": f"{data.pretty(f)} minus the cluster median"})
    return cols


def _cluster_card(row) -> dmc.Paper:
    tags = [dmc.Badge(t.replace("_", " "), variant="light", color="red" if t.startswith("high") else "teal",
                      size="sm", tt="none")
            for t in row["description"].split(", ")]
    return dmc.Paper(
        dmc.Stack([
            dmc.Group([dmc.Title(f"Cluster {row['cluster_id']}", order=5),
                       dmc.Badge(f"{row['n_states']} states", variant="outline")], justify="space-between"),
            dmc.Group(tags, gap=4),
            dmc.Text(row["states"].title(), size="xs", c="dimmed", lineClamp=3),
        ], gap="xs"),
        withBorder=True, p="md", radius="md",
    )


def layout(**_):
    if not data.pipeline_has_run():
        return pipeline_missing()

    profiles = data.cluster_profiles()
    silhouette = data.pipeline_metrics()["clustering"]["selected_silhouette"]
    states = data.state_names()

    return dmc.Container(
        [
            page_header(
                "Peer Clusters",
                f"KMeans (Spark ML) on each state's budget structure over the last 5 years. "
                f"{len(profiles)} clusters, silhouette {silhouette}. 'vs peers' = state value minus its cluster median.",
            ),
            dmc.SimpleGrid([_cluster_card(r) for _, r in profiles.iterrows()],
                           cols={"base": 1, "sm": 2, "lg": 4}, mb="md"),
            section(
                "Peer benchmark table",
                dmc.Stack([
                    dmc.Group(
                        [
                            dmc.Select(
                                id="cluster-filter",
                                label="Cluster",
                                data=[{"value": "all", "label": "All clusters"}] + [
                                    {"value": str(c), "label": f"Cluster {c}"} for c in profiles["cluster_id"]
                                ],
                                value="all",
                                allowDeselect=False,
                                w=200,
                            ),
                            dmc.MultiSelect(
                                id="cluster-features",
                                label="Features",
                                data=[{"value": f, "label": data.pretty(f)} for f in FEATURES],
                                value=DEFAULT_FEATURES,
                                searchable=True,
                                style={"flex": 1, "minWidth": 260},
                            ),
                        ],
                        align="flex-end",
                    ),
                    data_grid("cluster-grid", data.state_clusters(), _column_defs(DEFAULT_FEATURES)),
                ]),
                description="Green = more than 2 pp above peers, red = more than 2 pp below.",
            ),
            section(
                "How one state differs from its peers",
                dmc.Stack([
                    dmc.Select(id="cluster-state", label="State", data=states, value="CALIFORNIA"
                               if "CALIFORNIA" in states else states[0], searchable=True,
                               allowDeselect=False, w=260),
                    dmc.Text(id="cluster-peers", size="sm"),
                    dmc.BarChart(
                        id="cluster-gap-bar",
                        h=380,
                        dataKey="feature",
                        data=[],
                        orientation="vertical",
                        series=[{"name": "gap", "label": "Difference vs peer median", "color": "indigo.6"}],
                        valueFormatter={"function": "pctPoints"},
                        getBarColor={"function": "signColor"},
                        yAxisProps={"width": 190},
                    ),
                ]),
            ),
        ],
        fluid=True,
    )


@callback(
    Output("cluster-grid", "rowData"),
    Output("cluster-grid", "columnDefs"),
    Input("cluster-filter", "value"),
    Input("cluster-features", "value"),
)
def update_grid(cluster, features):
    df = data.state_clusters()
    if cluster != "all":
        df = df[df["cluster_id"] == int(cluster)]
    return to_records(df), _column_defs(features or [])


@callback(
    Output("cluster-gap-bar", "data"),
    Output("cluster-peers", "children"),
    Input("cluster-state", "value"),
)
def update_state_gaps(state):
    df = data.state_clusters()
    row = df[df["state"] == state].iloc[0]
    peers = sorted(df[(df["cluster_id"] == row["cluster_id"]) & (df["state"] != state)]["state"])
    gaps = [{"feature": data.pretty(f), "gap": float(row[f"{f}_vs_peers"])} for f in FEATURES]
    text = f"Cluster {row['cluster_id']} peers ({len(peers)}): " + ", ".join(p.title() for p in peers)
    return gaps, text
