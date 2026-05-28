# auto-generated: lowscan final selection helper
# purpose:
#   selected = good buy rule pass AND stop-before-target avoid fail AND no-bounce avoid fail

import numpy as np

import lowscan_good_buy_rules_v4_formatted as good_rules
import lowscan_stop_before_target_7_rules_formatted as stop_avoid_rules
import lowscan_target0_highprob_rules_cov4_c3_06_formatted as no_bounce_avoid_rules


def _rule_names_from_conditions(conditions, df, sep=","):
    names = []
    for i in range(len(df)):
        matched = [
            name
            for name, cond in conditions.items()
            if bool(cond.iloc[i] if hasattr(cond, "iloc") else cond[i])
        ]
        names.append(sep.join(matched))
    return names


def build_masks(df):
    good_conditions = good_rules.build_conditions(df)
    stop_conditions = stop_avoid_rules.build_conditions(df)
    no_bounce_conditions = no_bounce_avoid_rules.build_conditions(df)

    good_mask = np.zeros(len(df), dtype=bool)
    stop_avoid_mask = np.zeros(len(df), dtype=bool)
    no_bounce_avoid_mask = np.zeros(len(df), dtype=bool)

    for cond in good_conditions.values():
        good_mask |= cond

    for cond in stop_conditions.values():
        stop_avoid_mask |= cond

    for cond in no_bounce_conditions.values():
        no_bounce_avoid_mask |= cond

    avoid_mask = stop_avoid_mask | no_bounce_avoid_mask
    selected_mask = good_mask & (~avoid_mask)

    return {
        "good_mask": good_mask,
        "stop_avoid_mask": stop_avoid_mask,
        "no_bounce_avoid_mask": no_bounce_avoid_mask,
        "avoid_mask": avoid_mask,
        "selected_mask": selected_mask,
        "good_conditions": good_conditions,
        "stop_conditions": stop_conditions,
        "no_bounce_conditions": no_bounce_conditions,
    }


def apply_rule_columns(df):
    out = df.copy()
    m = build_masks(out)

    out["good_rule_pass"] = m["good_mask"].astype(int)
    out["stop_avoid_pass"] = m["stop_avoid_mask"].astype(int)
    out["no_bounce_avoid_pass"] = m["no_bounce_avoid_mask"].astype(int)
    out["avoid_rule_pass"] = m["avoid_mask"].astype(int)
    out["selected"] = m["selected_mask"].astype(int)

    out["good_rule_name"] = _rule_names_from_conditions(m["good_conditions"], out)
    out["stop_avoid_rule_name"] = _rule_names_from_conditions(m["stop_conditions"], out)
    out["no_bounce_avoid_rule_name"] = _rule_names_from_conditions(m["no_bounce_conditions"], out)

    return out


def build_selected_mask(df):
    return build_masks(df)["selected_mask"]
