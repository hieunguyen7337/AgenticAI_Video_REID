from __future__ import annotations

from collections import OrderedDict


PA100K_ATTRIBUTES = [
    "A female pedestrian",
    "A pedestrian over the age of 60",
    "A pedestrian between the ages of 18 and 60",
    "A pedestrian under the age of 18",
    "A pedestrian seen from the front",
    "A pedestrian seen from the side",
    "A pedestrian seen from the back",
    "A pedestrian wearing a hat",
    "A pedestrian wearing glasses",
    "A pedestrian with a handbag",
    "A pedestrian with a shoulder bag",
    "A pedestrian with a backpack",
    "A pedestrian holding objects in front",
    "A pedestrian in short-sleeved upper wear",
    "A pedestrian in long-sleeved upper wear",
    "A pedestrian in stride upper wear",
    "A pedestrian in upper wear with a logo",
    "A pedestrian in plaid upper wear",
    "A pedestrian in splice upper wear",
    "A pedestrian in striped lower wear",
    "A pedestrian in patterned lower wear",
    "A pedestrian in a long coat",
    "A pedestrian in trousers",
    "A pedestrian in shorts",
    "A pedestrian in skirts and dresses",
    "A pedestrian wearing boots",
]


def build_score_mapping(scores: list[float]) -> OrderedDict[str, float]:
    return OrderedDict((attribute, float(score)) for attribute, score in zip(PA100K_ATTRIBUTES, scores))
