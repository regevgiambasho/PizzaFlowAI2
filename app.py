"""PizzaFlow AI - final specification implementation.

This is an explainable local decision-support prototype. It does not claim to
be Generative AI when no API key is configured.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="PizzaFlow AI", page_icon="🍕", layout="wide")

# Final operating assumptions from the project specification
ORDER_START = 11 * 60 + 45
KITCHEN_START = 12 * 60
KITCHEN_END = 25 * 60
SLA_MINUTES = 40
DAILY_PIZZA_CAPACITY = 500

DEMAND_BY_DAY = {
    "Regular day": 350,
    "Weak Sunday": 250,
    "Busy Thursday": 450,
    "City event Thursday": 600,
}
PEAK_SHARE = {
    "Regular day": 0.35,
    "Weak Sunday": 0.25,
    "Busy Thursday": 0.70,
    "City event Thursday": 0.85,
}
PEAK_START = 18 * 60
PEAK_END = 22 * 60
PEAK_RANGE_BY_DAY = {
    "Regular day": (18 * 60, 22 * 60),
    "Weak Sunday": (18 * 60, 22 * 60),
    "Busy Thursday": (18 * 60, 22 * 60),
    "City event Thursday": (17 * 60, 23 * 60),
}
PEAK_WINDOW_MINUTES = {
    "Regular day": 240,
    "Weak Sunday": 240,
    "Busy Thursday": 240,
    "City event Thursday": 360,
}

CHEFS_NORMAL = 3
CHEFS_TUESDAY = 2
PACKERS = 1
DRONE_LOADERS = 1
BATTERY_TECHNICIANS = 1
HOURLY_WAGES = {
    "Chef": 50,
    "Packer": 35,
    "Drone loader": 100,
    "Battery technician": 35,
}
SHIFT_HOURS = 13
DAILY_RENT = 9000 / 30
DAILY_ELECTRICITY = 3000 / 30
DAILY_FIXED_COST = DAILY_RENT + DAILY_ELECTRICITY
TAX_RATE = 0.30

OVEN_CHAMBERS = {"Oven A": 3, "Oven B": 2, "Oven C": 1}
OVEN_C_WARMUP = 5
OVEN_C_COST_PER_PIZZA = 10
DRONE_COUNT = 5
DRONE_SPEED_KMH = 70
DRONE_RANGE_KM = 15
DRONE_FLIGHTS_BEFORE_BATTERY = 5
DRONE_BATTERY_MINUTES = 5
SECOND_DRONE_SURCHARGE = 50

PIZZA_PRICE = 60
TOPPING_PRICE = 10
DRINK_PRICE = {"None": 0, "Can": 10, "Large Bottle": 15}
DOUGH_COST = 4
CHEESE_SAUCE_COST = 10
TOPPING_COST = 2
DRINK_COST = {"None": 0, "Can": 2, "Large Bottle": 6}


def money(value: float) -> str:
    return f"₪{value:,.0f}"


def compensation(delivery_minutes: float, revenue: float) -> tuple[float, str]:
    """Compensation measured against the 40-minute SLA."""
    if delivery_minutes <= 40:
        return 0.0, "None"
    if delivery_minutes <= 60:
        return 10.0, "10 ₪ compensation"
    if delivery_minutes <= 70:
        return 30.0, "30 ₪ compensation"
    return float(revenue), "Full refund"


def order_row(
    order_id: int,
    arrival_minute: int,
    rng: random.Random,
    pizza_count: int | None = None,
    toppings: int | None = None,
    distance: int | None = None,
    drink: str | None = None,
) -> dict[str, Any]:
    pizza_count = pizza_count if pizza_count is not None else rng.randint(1, 5)
    toppings = toppings if toppings is not None else rng.randint(0, 3)
    distance = distance if distance is not None else rng.randint(1, DRONE_RANGE_KM)
    drink = drink if drink is not None else rng.choice(list(DRINK_PRICE))
    revenue = pizza_count * (PIZZA_PRICE + toppings * TOPPING_PRICE) + DRINK_PRICE[drink]
    materials = (
        pizza_count * (DOUGH_COST + CHEESE_SAUCE_COST + toppings * TOPPING_COST)
        + DRINK_COST[drink]
    )
    preparation = 2 + toppings
    baking = 7
    packaging = 1
    flight = max(1, math.ceil(distance / DRONE_SPEED_KMH * 60))
    estimated_eta = preparation + baking + packaging + flight
    return {
        "OrderID": order_id,
        "ArrivalMinute": arrival_minute,
        "PizzaCount": pizza_count,
        "Toppings": toppings,
        "Drink": drink,
        "DistanceKM": distance,
        "Revenue": revenue,
        "MaterialCost": materials,
        "BaseProfit": revenue - materials,
        "PreparationMinutes": preparation,
        "BakingMinutes": baking,
        "PackagingMinutes": packaging,
        "FlightMinutes": flight,
        "EstimatedETA": estimated_eta,
        "PromisedMinute": arrival_minute + SLA_MINUTES,
    }


def build_orders(day_type: str, tuesday: bool, seed: int = 42) -> tuple[list[dict[str, Any]], int]:
    requested = DEMAND_BY_DAY[day_type]
    accepted_target = min(requested, DAILY_PIZZA_CAPACITY)
    rng = random.Random(seed + list(DEMAND_BY_DAY).index(day_type))
    orders = []
    accepted_pizzas = 0
    order_id = 1
    # Demand is measured in pizzas, not orders. The last order is clipped so
    # that accepted inventory is exactly 250/350/450/500 pizzas as specified.
    while accepted_pizzas < accepted_target:
        remaining = accepted_target - accepted_pizzas
        pizza_count = min(rng.randint(1, 5), remaining)
        if rng.random() < PEAK_SHARE[day_type]:
            peak_start, peak_end = PEAK_RANGE_BY_DAY[day_type]
            arrival = rng.randint(
                peak_start,
                peak_end - 1,
            )
        else:
            arrival = KITCHEN_START + rng.randint(0, KITCHEN_END - KITCHEN_START - 30)
        orders.append(order_row(order_id, arrival, rng, pizza_count=pizza_count))
        accepted_pizzas += pizza_count
        order_id += 1
    return orders, requested - accepted_pizzas


def priority_score(order: dict[str, Any]) -> float:
    return round(order["BaseProfit"] / max(1, order["EstimatedETA"]), 3)


def fifo_ids(orders: list[dict[str, Any]]) -> list[int]:
    return [int(row["OrderID"]) for row in sorted(orders, key=lambda x: (x["ArrivalMinute"], x["OrderID"]))]


def local_ai_ids(orders: list[dict[str, Any]]) -> list[int]:
    # The transparent local fallback follows Profit / ETA within five-minute
    # arrival windows. This keeps the dispatch policy responsive to newly
    # arrived orders instead of scheduling a future order ahead of an already
    # waiting customer and breaking the SLA.
    return [
        int(row["OrderID"])
        for row in sorted(
            orders,
            key=lambda x: (x["ArrivalMinute"] // 5, -priority_score(x), x["OrderID"]),
        )
    ]


def optimizer_candidates(orders: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    """Generate transparent schedules for the local optimizer to evaluate."""
    return [
        ("FIFO", fifo_ids(orders)),
        ("Profit / ETA (5-minute window)", local_ai_ids(orders)),
        (
            "Profit / ETA (15-minute window)",
            [
                int(row["OrderID"])
                for row in sorted(
                    orders,
                    key=lambda x: (x["ArrivalMinute"] // 15, -priority_score(x), x["OrderID"]),
                )
            ],
        ),
        (
            "Urgency with profit tie-break",
            [
                int(row["OrderID"])
                for row in sorted(
                    orders,
                    key=lambda x: (x["PromisedMinute"], -priority_score(x), x["OrderID"]),
                )
            ],
        ),
    ]


def labor_cost(tuesday: bool) -> float:
    chefs = CHEFS_TUESDAY if tuesday else CHEFS_NORMAL
    return (
        chefs * HOURLY_WAGES["Chef"] * SHIFT_HOURS
        + PACKERS * HOURLY_WAGES["Packer"] * SHIFT_HOURS
        + DRONE_LOADERS * HOURLY_WAGES["Drone loader"] * SHIFT_HOURS
        + BATTERY_TECHNICIANS * HOURLY_WAGES["Battery technician"] * SHIFT_HOURS
    )


def simulate(orders: list[dict[str, Any]], sequence: list[int], tuesday: bool) -> dict[str, Any]:
    by_id = {int(row["OrderID"]): row for row in orders}
    chefs = CHEFS_TUESDAY if tuesday else CHEFS_NORMAL
    chef_available = [KITCHEN_START] * chefs
    packer_available = [KITCHEN_START] * PACKERS
    loader_available = [KITCHEN_START] * DRONE_LOADERS
    ovens = []
    for name, count in OVEN_CHAMBERS.items():
        for slot in range(count):
            ovens.append({"name": name, "available": KITCHEN_START, "c_slot": slot})
    drones = [{"available": KITCHEN_START, "flights": 0} for _ in range(DRONE_COUNT)]
    details: list[dict[str, Any]] = []

    for position, order_id in enumerate(sequence, start=1):
        order = by_id[int(order_id)]
        chef_index = min(range(len(chef_available)), key=lambda i: chef_available[i])
        prep_start = max(order["ArrivalMinute"], chef_available[chef_index])
        prep_finish = prep_start + order["PreparationMinutes"]
        chef_available[chef_index] = prep_finish

        # Select the oven that finishes earliest. Oven C includes its warm-up
        # when it is needed and incurs its stated per-pizza cost.
        # A chamber can bake several pizzas from the same order in one batch;
        # each additional pizza adds handling/load time without multiplying the
        # full seven-minute bake cycle.
        baking_duration = order["BakingMinutes"] + max(0, order["PizzaCount"] - 1) * 2
        oven_options = []
        for oven in ovens:
            start = max(prep_finish, oven["available"])
            warmup = OVEN_C_WARMUP if oven["name"] == "Oven C" else 0
            oven_options.append((start + warmup + baking_duration, start, warmup, oven))
        finish, oven_start, warmup, oven = min(oven_options, key=lambda x: (x[0], x[3]["name"]))
        oven["available"] = finish
        baking_finish = finish

        packer_index = min(range(len(packer_available)), key=lambda i: packer_available[i])
        package_start = max(baking_finish, packer_available[packer_index])
        package_finish = package_start + order["PackagingMinutes"]
        packer_available[packer_index] = package_finish

        drone_index = min(range(len(drones)), key=lambda i: drones[i]["available"])
        drone = drones[drone_index]
        battery_break = DRONE_BATTERY_MINUTES if drone["flights"] >= DRONE_FLIGHTS_BEFORE_BATTERY else 0
        flight_start = max(package_finish, drone["available"]) + battery_break
        delivered = flight_start + order["FlightMinutes"]
        drone["available"] = delivered
        drone["flights"] = 0 if drone["flights"] >= DRONE_FLIGHTS_BEFORE_BATTERY else drone["flights"] + 1

        elapsed = delivered - order["ArrivalMinute"]
        drone_surcharge = SECOND_DRONE_SURCHARGE if drone_index > 0 else 0
        charged_revenue = order["Revenue"] + drone_surcharge
        refund, refund_label = compensation(elapsed, charged_revenue)
        oven_c_cost = OVEN_C_COST_PER_PIZZA * order["PizzaCount"] if oven["name"] == "Oven C" else 0
        realized = charged_revenue - order["MaterialCost"] - refund - oven_c_cost
        details.append(
            {
                "Sequence": position,
                "OrderID": order["OrderID"],
                "DeliveredMinute": delivered,
                "ActualETA": elapsed,
                "PromisedMinute": order["PromisedMinute"],
                "LateMinutes": max(0, elapsed - SLA_MINUTES),
                "OnTime": "Yes" if elapsed <= SLA_MINUTES else "No",
                "RefundOrCompensation": refund_label,
                "CompensationCost": refund,
                "DroneNumber": drone_index + 1,
                "DroneSurcharge": drone_surcharge,
                "ChargedRevenue": charged_revenue,
                "Oven": oven["name"],
                "OvenCCost": oven_c_cost,
                "RealizedContribution": realized,
            }
        )

    details_df = pd.DataFrame(details)
    contribution = float(details_df["RealizedContribution"].sum()) if not details_df.empty else 0.0
    total_comp = float(details_df["CompensationCost"].sum()) if not details_df.empty else 0.0
    drone_surcharge_total = float(details_df["DroneSurcharge"].sum()) if not details_df.empty else 0.0
    full_refunds = int((details_df["RefundOrCompensation"] == "Full refund").sum()) if not details_df.empty else 0
    labor = labor_cost(tuesday)
    pre_tax = contribution - labor - DAILY_FIXED_COST
    adjusted_net = pre_tax * (1 - TAX_RATE) if pre_tax > 0 else pre_tax
    return {
        "details": details_df,
        "contribution": contribution,
        "total_compensation_cost": total_comp,
        "drone_surcharge_total": drone_surcharge_total,
        "labor": labor,
        "fixed": DAILY_FIXED_COST,
        "pre_tax": pre_tax,
        "adjusted_net": adjusted_net,
        "avg_eta": float(details_df["ActualETA"].mean()) if not details_df.empty else 0,
        "on_time_rate": float((details_df["OnTime"] == "Yes").mean() * 100) if not details_df.empty else 0,
        "compensation_rate": float((details_df["CompensationCost"] > 0).mean() * 100) if not details_df.empty else 0,
        "full_refund_rate": full_refunds / len(details_df) * 100 if not details_df.empty else 0,
        "full_refunds": full_refunds,
    }


def run_evaluation(orders: list[dict[str, Any]], tuesday: bool) -> dict[str, Any]:
    candidates = []
    for label, sequence in optimizer_candidates(orders):
        candidates.append((label, sequence, simulate(orders, sequence, tuesday)))
    fifo = next(result for label, sequence, result in candidates if label == "FIFO")

    # The local optimizer evaluates the same orders and resources under every
    # candidate schedule. SLA protection is the first decision criterion;
    # adjusted net profit is used as the tie-breaker. FIFO is always included,
    # so the AI cannot claim an SLA improvement by making service worse.
    selected_label, ai_sequence, ai = max(
        candidates,
        key=lambda item: (
            item[2]["on_time_rate"],
            item[2]["adjusted_net"],
            -item[2]["compensation_rate"],
            -item[2]["avg_eta"],
        ),
    )
    safety_fallback = selected_label == "FIFO"
    profit_improvement = ((ai["adjusted_net"] - fifo["adjusted_net"]) / abs(fifo["adjusted_net"]) * 100) if fifo["adjusted_net"] else 0
    eta_improvement = ((fifo["avg_eta"] - ai["avg_eta"]) / fifo["avg_eta"] * 100) if fifo["avg_eta"] else 0
    on_time_advantage = ai["on_time_rate"] - fifo["on_time_rate"]
    return {
        "fifo": fifo,
        "ai": ai,
        "profit_improvement": profit_improvement,
        "eta_improvement": eta_improvement,
        "on_time_advantage": on_time_advantage,
        "sequence": ai_sequence,
        "safety_fallback": safety_fallback,
        "selected_label": selected_label,
    }


def kpi_status(value: float, target: float, higher_is_better: bool = True) -> str:
    return "✅" if (value >= target if higher_is_better else value <= target) else "⚠️"


if "orders" not in st.session_state:
    st.session_state.orders = []
if "day_type" not in st.session_state:
    st.session_state.day_type = "Regular day"
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "tuesday" not in st.session_state:
    st.session_state.tuesday = False


st.title("🍕 PizzaFlow AI")
st.caption("Explainable decision-support for a dark kitchen with autonomous drone delivery")
st.info("Local transparent engine active. It does not claim to be Generative AI without an API connection.")

tab_orders, tab_dashboard, tab_method = st.tabs(["🍕 Orders", "📊 Dashboard", "🛡 Method & Responsible AI"])

with tab_orders:
    st.header("Create Order")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pizzas = st.number_input("Number of pizzas", 1, 10, 1)
    with c2:
        toppings = st.number_input("Toppings", 0, 3, 0)
    with c3:
        drink = st.selectbox("Drink", list(DRINK_PRICE))
    with c4:
        distance = st.slider("Distance (km)", 1, 15, 5)
    if st.button("🍕 Place Order"):
        next_id = max([int(x["OrderID"]) for x in st.session_state.orders], default=0) + 1
        st.session_state.orders.append(
            order_row(
                next_id,
                KITCHEN_START,
                random.Random(next_id),
                pizza_count=int(pizzas),
                toppings=int(toppings),
                distance=int(distance),
                drink=drink,
            )
        )
        st.success(f"Order #{next_id} created.")
    if st.session_state.orders:
        st.dataframe(pd.DataFrame(st.session_state.orders), use_container_width=True)

with tab_dashboard:
    st.header("Operations Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        day_type = st.selectbox("Day scenario", list(DEMAND_BY_DAY), key="day_type_select")
    with c2:
        tuesday = st.checkbox("Tuesday: 2 chefs", key="tuesday_select")
    with c3:
        if st.button("🎬 Load Scenario"):
            st.session_state.orders, rejected = build_orders(day_type, tuesday)
            st.session_state.day_type = day_type
            st.session_state.tuesday = tuesday
            st.session_state.evaluation = None
            accepted_pizzas = sum(int(x["PizzaCount"]) for x in st.session_state.orders)
            st.success(f"Loaded {accepted_pizzas} accepted pizzas in {len(st.session_state.orders)} orders. Rejected pizzas after capacity: {rejected}.")
    with c4:
        if st.button("🗑 Reset"):
            st.session_state.orders = []
            st.session_state.evaluation = None
            st.rerun()

    if not st.session_state.orders:
        st.warning("Choose a scenario and click Load Scenario.")
    else:
        requested = DEMAND_BY_DAY[st.session_state.day_type]
        accepted_pizzas = sum(int(x["PizzaCount"]) for x in st.session_state.orders)
        rejected = max(0, requested - accepted_pizzas)
        st.write(f"Scenario: **{st.session_state.day_type}** | Requested pizzas: **{requested}** | Accepted pizzas: **{accepted_pizzas}** | Rejected pizzas: **{rejected}** | Orders: **{len(st.session_state.orders)}**")
        display = pd.DataFrame(st.session_state.orders).copy()
        display["Priority"] = display.apply(priority_score, axis=1)
        st.dataframe(display, use_container_width=True, height=360)
        if st.button("🤖 Analyze & Compare", type="primary"):
            st.session_state.evaluation = run_evaluation(st.session_state.orders, st.session_state.tuesday)

        result = st.session_state.evaluation
        if result:
            fifo = result["fifo"]
            ai = result["ai"]
            stress_test = st.session_state.day_type == "City event Thursday"
            if stress_test:
                st.warning(
                    "Stress Test: this scenario intentionally exceeds daily demand capacity. "
                    "Profit-improvement KPI is not treated as a normal performance result."
                )
            st.subheader("KPI Dashboard")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("AI on-time", f"{ai['on_time_rate']:.1f}%", delta_color="off")
            k1.caption(f"{kpi_status(ai['on_time_rate'], 98)} יעד: 98%")
            k2.metric(
                "Profit improvement",
                "Stress test" if stress_test else f"{result['profit_improvement']:.1f}%",
                delta_color="off",
            )
            k2.caption("⚪ לא נמדד כיעד רגיל" if stress_test else f"{kpi_status(result['profit_improvement'], 15)} יעד: 15%")
            k3.metric("Delivery improvement", f"{result['eta_improvement']:.1f}%", delta_color="off")
            k3.caption(f"{kpi_status(result['eta_improvement'], 10)} יעד: 10%")
            k4.metric("Compensation", f"{ai['compensation_rate']:.2f}%", delta_color="off")
            k4.caption(f"{kpi_status(ai['compensation_rate'], 1, False)} מקסימום: 1%")
            k5.metric("Full refunds", f"{ai['full_refund_rate']:.2f}%", delta_color="off")
            k5.caption(f"{kpi_status(ai['full_refund_rate'], 0.1, False)} מקסימום: 0.1%")

            if result["on_time_advantage"] > 0:
                st.success(
                    f"🚀 יתרון PizzaFlow AI מול FIFO: "
                    f"+{result['on_time_advantage']:.1f} נקודות אחוז בהזמנות שסופקו בתוך 40 דקות."
                )
            elif result["on_time_advantage"] < 0:
                st.warning(
                    f"FIFO השיג יתרון של {abs(result['on_time_advantage']):.1f} נקודות אחוז בעמידה ב־SLA בתרחיש זה."
                )
            else:
                st.info("PizzaFlow AI ו־FIFO השיגו אותה עמידה ב־SLA בתרחיש זה.")

            st.subheader("FIFO vs PizzaFlow AI")
            if result["safety_fallback"]:
                st.info("The optimizer evaluated several schedules and selected FIFO because it was the best available schedule for this scenario.")
            else:
                st.success(f"The optimizer selected: {result['selected_label']}")
            comparison = pd.DataFrame({
                "Strategy": ["FIFO", "PizzaFlow AI"],
                "Adjusted Net Profit": [fifo["adjusted_net"], ai["adjusted_net"]],
                "Average ETA": [fifo["avg_eta"], ai["avg_eta"]],
                "On-time rate": [fifo["on_time_rate"], ai["on_time_rate"]],
            })
            st.dataframe(comparison, use_container_width=True)
            fig = px.bar(comparison, x="Strategy", y="Adjusted Net Profit", color="Strategy", title="Adjusted Net Profit Comparison")
            st.plotly_chart(fig, use_container_width=True)

            best_order = display.sort_values("Priority", ascending=False).iloc[0]
            st.success(
                f"🤖 Local explainable engine selected Order #{int(best_order['OrderID'])}. "
                f"Priority = Profit / ETA = {best_order['Priority']:.3f}."
            )
            st.write(
                f"Adjusted net profit: {money(ai['adjusted_net'])} | "
                f"Contribution after delivery costs: {money(ai['contribution'])} | "
                f"Drone surcharges: {money(ai['drone_surcharge_total'])} | "
                f"Compensation/refunds: {money(ai['total_compensation_cost'])} | "
                f"Labor: {money(ai['labor'])} | Fixed rent/electricity: {money(ai['fixed'])} | "
                f"Tax rate: {TAX_RATE:.0%}"
            )
            st.subheader("Business Outcome")
            st.dataframe(ai["details"].head(30), use_container_width=True)

with tab_method:
    st.header("Method & Responsible AI")
    st.markdown("""
**Objective:** Maximize Adjusted Net Profit subject to SLA ≤ 40 minutes, daily capacity of 500 pizzas, chef capacity, oven capacity, and drone capacity.

**Priority:** `Profit / ETA`, where ETA = preparation + baking + packaging + flight.

**Resources:** 3 chefs normally (2 on Tuesday), 1 packer, 1 drone loader, 1 battery technician, Oven A with 3 chambers, Oven B with 2, and Oven C with 1 chamber that needs 5 minutes warm-up and costs 10 ₪ per pizza.

**Responsible AI:** The current deployment is a transparent local fallback. It uses only the supplied order fields, does not invent order IDs, and exposes its assumptions. A human remains responsible for operational decisions.
""")
    st.subheader("Operating assumptions")
    st.json({
        "SLA_minutes": SLA_MINUTES,
        "daily_pizza_capacity": DAILY_PIZZA_CAPACITY,
        "drone_count": DRONE_COUNT,
        "drone_speed_kmh": DRONE_SPEED_KMH,
        "drone_range_km": DRONE_RANGE_KM,
        "daily_fixed_cost": DAILY_FIXED_COST,
        "tax_rate": TAX_RATE,
    })
