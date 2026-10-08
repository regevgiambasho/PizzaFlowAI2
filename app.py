"""PizzaFlow AI - final specification implementation.

This is an explainable local decision-support prototype. It includes a local,
grounded manager assistant so the public deployment does not require an API
key, an external service, or a paid model.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime
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
    "Regular day": 0.45,
    "Weak Sunday": 0.25,
    "Busy Thursday": 0.70,
    # An exceptional event still concentrates demand, but the remaining
    # orders arrive across the operating day so the kitchen can protect the
    # 40-minute SLA instead of modelling an impossible six-hour spike.
    "City event Thursday": 0.65,
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
PEAK_SURCHARGE_RATE = {
    "Regular day": 0.00,
    "Weak Sunday": 0.00,
    "Busy Thursday": 0.10,
    "City event Thursday": 0.15,
}
EVENT_EXTRA_CHEFS = 1
PEAK_EXTRA_OVEN_CHAMBERS = {
    "Regular day": 0,
    "Weak Sunday": 0,
    "Busy Thursday": 2,
    "City event Thursday": 0,
}
PEAK_EXTRA_PACKERS = {
    "Regular day": 0,
    "Weak Sunday": 0,
    "Busy Thursday": 1,
    "City event Thursday": 0,
}

PIZZA_PRICE = 60
TOPPING_PRICE = 10
DRINK_PRICE = {"None": 0, "Can": 10, "Large Bottle": 15}
DOUGH_COST = 4
CHEESE_SAUCE_COST = 10
TOPPING_COST = 2
DRINK_COST = {"None": 0, "Can": 2, "Large Bottle": 6}


def money(value: float) -> str:
    return f"₪{value:,.0f}"


def normalize_question(question: str) -> str:
    """Normalize Hebrew/English text for deterministic local intent matching."""
    return re.sub(r"[^\w\s/%+-]", " ", question.lower()).strip()


def scenario_description(day_type: str) -> str:
    """Return the demand assumptions for the selected scenario."""
    requested = DEMAND_BY_DAY[day_type]
    if day_type == "Regular day":
        return "יום רגיל: 350 פיצות מבוקשות, ללא תוספת משאבים או תוספת מחיר."
    if day_type == "Weak Sunday":
        return "יום ראשון חלש: 250 פיצות מבוקשות, ללא תוספת משאבים או תוספת מחיר."
    if day_type == "Busy Thursday":
        return (
            "יום חמישי עמוס: 450 פיצות מבוקשות. מדיניות השיא יכולה להפעיל 2 תאי תנור "
            "זמניים, עובד אריזה נוסף ותוספת מחיר של 10%, כדי להגן על ה־SLA ולממן "
            "את קיבולת השיא."
        )
    return (
        f"אירוע עירוני: {requested} פיצות מבוקשות, אך המלאי היומי מוגבל ל־"
        f"{DAILY_PIZZA_CAPACITY}; לכן 100 פיצות נדחות והמערכת משווה בין שתי דרכי בחירה "
        "של 500 הפיצות שאפשר לייצר. מדיניות האירוע מוסיפה טבח זמני ותוספת מחיר של 15%."
    )


def current_result_answer(day_type: str, result: dict[str, Any]) -> str:
    """Explain only metrics that were actually calculated by the simulation."""
    fifo = result["fifo"]
    ai = result["ai"]
    selected = result["selected_label"]
    improvement = result["profit_improvement"]
    improvement_text = (
        f"שיפור של {improvement:.1f}% ברווח הנקי המתואם"
        if fifo["adjusted_net"] > 0
        else f"פער של {money(result['profit_delta'])} מול בסיס FIFO"
    )
    return (
        f"**תוצאות {day_type}:**\n\n"
        f"• PizzaFlow: רווח נקי מתואם {money(ai['adjusted_net'])}, "
        f"עמידה ב־SLA של {ai['on_time_rate']:.1f}%, זמן ממוצע {ai['avg_eta']:.1f} דקות.\n"
        f"• FIFO: רווח נקי מתואם {money(fifo['adjusted_net'])}, "
        f"עמידה ב־SLA של {fifo['on_time_rate']:.1f}%, זמן ממוצע {fifo['avg_eta']:.1f} דקות.\n"
        f"• ההשוואה: {improvement_text}; שיפור זמן האספקה הוא {result['eta_improvement']:.1f}%.\n"
        f"• פיצויים ב־PizzaFlow: {ai['compensation_rate']:.2f}% מההזמנות; "
        f"החזרים מלאים: {ai['full_refund_rate']:.2f}%.\n"
        f"• מדיניות התזמון שנבחרה: **{selected}**.\n\n"
        "המלצה: להשתמש בתוצאה כבסיס להחלטה, אבל לבדוק אנושית את עלות התוספת "
        "והאם הביקוש בפועל דומה להנחות הסימולציה."
    )


def infer_day_type(question: str, fallback: str) -> str:
    """Infer a scenario named in the question, while keeping the UI choice as fallback."""
    if any(marker in question for marker in ("אירוע", "600", "city event", "event")):
        return "City event Thursday"
    if any(marker in question for marker in ("חמישי", "450", "busy thursday", "busy")):
        return "Busy Thursday"
    if any(marker in question for marker in ("ראשון", "250", "weak sunday", "sunday")):
        return "Weak Sunday"
    if any(marker in question for marker in ("רגיל", "350", "regular day", "regular")):
        return "Regular day"
    return fallback


def local_manager_answer(
    question: str,
    day_type: str,
    result: dict[str, Any] | None = None,
) -> str:
    """Answer manager questions locally from the project specification.

    This is intentionally not presented as a general-purpose language model.
    It uses intent matching and grounded response templates, so it remains
    free, deterministic, auditable, and safe for a public demo.
    """
    question = question.strip()
    normalized = normalize_question(question)
    if not normalized:
        return "כתוב שאלה על המערכת, על התרחיש או על תוצאות ההשוואה."
    question_day_type = infer_day_type(normalized, day_type)

    intent_keywords = {
        "results": [
            "תוצאה", "השוואה", "fifo", "שיפור", "כמה הרווח", "כמה יצא",
            "כדאי", "המלצה", "on time", "עמידה", "performance", "compare",
        ],
        "objective": ["מטרה", "objective", "למקסם", "רווחיות", "adjusted net profit"],
        "algorithm": [
            "אלגוריתם", "תיעדוף", "priority", "priorit", "profit / eta",
            "profit eta", "נוסחה", "דירוג", "איך נבחרת הזמנה",
        ],
        "ai_role": [
            "מה התפקיד של ai", "תפקיד ה ai", "תפקיד ai", "מה עושה ai", "בינה מלאכותית",
            "רכיב ai", "מנוע ai", "artificial intelligence", "איך ai עובד", "גנרטיבי",
            "generative",
        ],
        "timing": [
            "זמן", "eta", "sla", "40", "הכנה", "אפייה", "אריזה", "טיסה",
            "איחור", "delivery", "דקות",
        ],
        "capacity": [
            "מלאי", "500", "capacity", "סגירת", "נדחות", "דחייה", "כמה אפשר",
            "קיבולת",
        ],
        "staff": [
            "עובד", "עובדים", "טבח", "טבחים", "שכר", "chef", "packer",
            "loader", "טכנאי", "משמרת", "שלישי",
        ],
        "ovens": ["תנור", "תנורים", "תאים", "oven", "חימום", "chamber"],
        "drones": [
            "רחפן", "רחפנים", "drone", "סוללה", "70", "15 קמ", "טיסות",
            "מהירות",
        ],
        "pricing": [
            "מחיר", "עלות", "תוספת", "שתייה", "פחית", "בקבוק", "מס", "שכירות",
            "חשמל", "הכנסה", "revenue", "cost",
        ],
        "scenarios": [
            "רגיל", "חלש", "ראשון", "חמישי", "אירוע", "350", "250", "450", "600",
            "scenario",
        ],
        "compensation": [
            "פיצוי", "החזר", "refund", "compensation", "איחור", "45", "60", "70",
        ],
        "kpi": ["kpi", "98", "15%", "10%", "1%", "0.1", "יעד"],
        "responsible": [
            "אחריות", "סיכון", "responsible", "הטיה", "מגבלה", "אדם", "אנושי",
            "הסבר", "explain",
        ],
        "specification": [
            "הגדרות", "מפרט", "מערכת", "specification", "assumption", "הנחות",
        ],
    }
    scores = {
        intent: sum(1 for keyword in keywords if keyword in normalized)
        for intent, keywords in intent_keywords.items()
    }

    comparison_markers = (
        "fifo", "השוואה", "שיפור", "תוצאה", "כמה יצא", "כדאי", "יתרון", "רווח",
        "האם כדאי", "ביצועים",
    )
    if result is not None and any(marker in normalized for marker in comparison_markers):
        return current_result_answer(day_type, result)

    best_intent, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return (
            "אני המנוע המקומי של PizzaFlow AI. אני יכול לענות על: מטרת המערכת, "
            "האלגוריתם, SLA וזמני ייצור, עובדים ושכר, תנורים, רחפנים, מלאי, "
            "תמחור, תרחישים, KPI, פיצויים ו־Responsible AI. "
            "נסה לנסח את השאלה באחד מהנושאים האלה."
        )

    if best_intent == "objective":
        return (
            "מטרת PizzaFlow AI היא למקסם **Adjusted Net Profit** תוך עמידה ב־SLA "
            "של עד 40 דקות. המנוע מתחשב במלאי של 500 פיצות ביום, בכוח האדם, "
            "בקיבולת התנורים ובקיבולת הרחפנים."
        )
    if best_intent == "algorithm":
        return (
            "לכל הזמנה מחושבים Revenue, עלות חומרי גלם, Profit ו־ETA.\n\n"
            "**ETA = הכנה + אפייה + אריזה + טיסה**\n\n"
            "**Priority = Profit / ETA**\n\n"
            "המנוע משווה כמה רצפים שקופים, כולל FIFO, Profit/ETA ורצף המגן על ה־SLA. "
            "הקריטריון הראשון הוא עמידה ב־40 דקות; לאחר מכן נבחר הרווח הנקי המתואם הגבוה יותר."
        )
    if best_intent == "ai_role":
        return (
            "רכיב ה־AI המרכזי הוא מנוע החלטה מקומי: הוא מדרג הזמנות, בוחן קבלה "
            "כאשר הביקוש עובר 500 פיצות, משווה רצפי ייצור ומאזן בין רווח לבין זמן אספקה. "
            "בנוסף יש כאן עוזר שאלות מקומי שמחזיר תשובות grounded מהמפרט ומהסימולציה. "
            "אין שימוש ב־OpenAI, אין מפתח API ואין שליחת נתונים החוצה. זהו מנוע AI "
            "מסביר/היוריסטי, לא מודל שפה גנרטיבי כללי."
        )
    if best_intent == "timing":
        return (
            "זמן ההזמנה מחושב כך: הכנה = 2 דקות בסיס + דקה לכל תוספת; אפייה = 7 דקות; "
            "אריזה = דקה; טיסה מחושבת לפי המרחק ומהירות הרחפן. יעד ה־SLA הוא עד 40 דקות. "
            "המערכת בודקת את הזמן בפועל לאחר תורים אצל טבחים, תנורים, אריזה ורחפנים."
        )
    if best_intent == "capacity":
        return (
            f"המלאי היומי הוא {DAILY_PIZZA_CAPACITY} פיצות. ביום רגיל הביקוש הוא 350, "
            "בחמישי עמוס 450, ובאירוע עירוני 600. באירוע העירוני המערכת מקבלת רק "
            f"{DAILY_PIZZA_CAPACITY} פיצות ודוחה את 100 הנותרות; היא משווה בין FIFO "
            "לבין בחירה לפי ערך עסקי."
        )
    if best_intent == "staff":
        return (
            f"מצבת הבסיס היא {CHEFS_NORMAL} טבחים, עובד אריזה אחד, מעמיס רחפנים אחד "
            f"וטכנאי סוללות אחד. ביום שלישי יש {CHEFS_TUESDAY} טבחים. המשמרת היא "
            f"12:00–01:00, כלומר {SHIFT_HOURS} שעות. השכר לשעה: טבח {HOURLY_WAGES['Chef']} ₪, "
            f"אריזה {HOURLY_WAGES['Packer']} ₪, מעמיס {HOURLY_WAGES['Drone loader']} ₪, "
            f"טכנאי {HOURLY_WAGES['Battery technician']} ₪."
        )
    if best_intent == "ovens":
        return (
            "Oven A כולל 3 תאים ו־Oven B כולל 2 תאים; שניהם פעילים תמיד. Oven C כולל תא אחד, "
            "דורש 5 דקות חימום ועולה 10 ₪ לכל פיצה, ולכן הוא מופעל רק כשמדיניות העומס "
            "מצדיקה זאת."
        )
    if best_intent == "drones":
        return (
            f"במערכת יש {DRONE_COUNT} רחפנים, במהירות {DRONE_SPEED_KMH} קמ״ש ובטווח של "
            f"{DRONE_RANGE_KM} ק״מ. אחרי {DRONE_FLIGHTS_BEFORE_BATTERY} טיסות רחפן מושבת "
            f"ל־{DRONE_BATTERY_MINUTES} דקות להחלפת סוללה. הרחפן הראשון חינם; רחפן נוסף "
            f"מחויב ב־{SECOND_DRONE_SURCHARGE} ₪ ללקוח."
        )
    if best_intent == "pricing":
        return (
            f"מחיר בסיס לפיצה הוא {PIZZA_PRICE} ₪, תוספת עולה {TOPPING_PRICE} ₪, פחית "
            f"{DRINK_PRICE['Can']} ₪ ובקבוק גדול {DRINK_PRICE['Large Bottle']} ₪. עלויות "
            f"החומרים הן בצק {DOUGH_COST} ₪, גבינה ורוטב {CHEESE_SAUCE_COST} ₪, ותוספת "
            f"{TOPPING_COST} ₪. בנוסף מחושבים שכר, שכירות, חשמל, מס של {TAX_RATE:.0%}, "
            "פיצויים ועלות Oven C."
        )
    if best_intent == "scenarios":
        return scenario_description(question_day_type)
    if best_intent == "compensation":
        return (
            "מדיניות הפיצוי מבוססת על ה־SLA של 40 דקות: עד 40 דקות אין פיצוי; "
            "בין 40 ל־60 דקות יש פיצוי של 10 ₪; בין 60 ל־70 דקות 30 ₪; מעל 70 דקות "
            "יש החזר מלא והלקוח שומר את ההזמנה. העלויות האלה נכנסות לחישוב Adjusted Net Profit."
        )
    if best_intent == "kpi":
        return (
            "ה־KPI שהוגדרו הם: 98% מההזמנות עד 40 דקות; שיפור רווח מול FIFO; "
            "שיפור של 10% בזמן האספקה; פחות מ־1% הזמנות עם פיצוי; ופחות מ־0.1% "
            "החזרים מלאים. בגרסה הנוכחית יעדי השיפור לתרחישים מוצגים בנפרד: קטן ביום רגיל, "
            "כ־10% בחמישי, ו־16%–20% באירוע עירוני."
        )
    if best_intent == "responsible":
        return (
            "הסיכונים העיקריים הם הסתמכות על הנחות סימולציה, הטיה לטובת הזמנות רווחיות, "
            "אי־דיוק בתחזית הביקוש, ושימוש בנתונים שאינם מייצגים. לכן המערכת מסבירה את "
            "הנוסחה וההנחות, לא מבצעת פעולה אמיתית, והחלטה אנושית נדרשת לפני שינוי תפעולי."
        )
    if best_intent == "specification":
        return (
            f"{scenario_description(day_type)} מטרת המערכת היא למקסם רווח נקי מתואם תחת "
            "SLA של 40 דקות, מלאי של 500 פיצות, מגבלות טבחים, תנורים ורחפנים. "
            "היא כוללת מסך הזמנות, תרחישים, Dashboard, השוואת FIFO, KPI, מנוע תיעדוף "
            "והסבר Responsible AI."
        )

    if result is not None:
        return current_result_answer(day_type, result)
    return scenario_description(question_day_type)


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


def adjust_order_pizzas(order: dict[str, Any], pizza_count: int) -> dict[str, Any]:
    """Return a clipped copy when stock capacity ends inside an order."""
    updated = dict(order)
    updated["PizzaCount"] = pizza_count
    updated["Revenue"] = pizza_count * (PIZZA_PRICE + order["Toppings"] * TOPPING_PRICE) + DRINK_PRICE[order["Drink"]]
    updated["MaterialCost"] = pizza_count * (DOUGH_COST + CHEESE_SAUCE_COST + order["Toppings"] * TOPPING_COST) + DRINK_COST[order["Drink"]]
    updated["BaseProfit"] = updated["Revenue"] - updated["MaterialCost"]
    return updated


def fifo_admission(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept the first orders until the daily pizza inventory is exhausted."""
    accepted: list[dict[str, Any]] = []
    accepted_pizzas = 0
    for order in sorted(orders, key=lambda x: (x["ArrivalMinute"], x["OrderID"])):
        remaining = DAILY_PIZZA_CAPACITY - accepted_pizzas
        if remaining <= 0:
            break
        count = min(int(order["PizzaCount"]), remaining)
        accepted.append(adjust_order_pizzas(order, count) if count != order["PizzaCount"] else order)
        accepted_pizzas += count
    return accepted


def ai_admission(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use stock on the highest-value orders while preserving arrival fairness."""
    accepted: list[dict[str, Any]] = []
    accepted_pizzas = 0
    ranked = sorted(
        orders,
        key=lambda x: (-priority_score(x), -x["BaseProfit"], x["ArrivalMinute"], x["OrderID"]),
    )
    for order in ranked:
        remaining = DAILY_PIZZA_CAPACITY - accepted_pizzas
        if remaining <= 0:
            break
        count = min(int(order["PizzaCount"]), remaining)
        accepted.append(adjust_order_pizzas(order, count) if count != order["PizzaCount"] else order)
        accepted_pizzas += count
    return accepted


def build_orders(day_type: str, tuesday: bool, seed: int = 42) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    requested = DEMAND_BY_DAY[day_type]
    rng = random.Random(seed + list(DEMAND_BY_DAY).index(day_type))
    orders = []
    requested_pizzas = 0
    order_id = 1
    # Generate all requested demand first. Admission is then evaluated
    # separately for FIFO and PizzaFlow AI when inventory is constrained.
    while requested_pizzas < requested:
        remaining = requested - requested_pizzas
        if day_type == "City event Thursday":
            # Event demand is heterogeneous: premium group orders, standard
            # orders, and low-margin/far deliveries. This gives the admission
            # optimizer a real economic decision when stock is limited.
            mix = rng.random()
            if mix < 0.35:
                pizza_count = rng.randint(3, 5)
                toppings = rng.randint(2, 3)
                drink = "Large Bottle"
                distance = rng.randint(1, 8)
            elif mix < 0.50:
                pizza_count = rng.randint(1, 4)
                toppings = rng.randint(0, 2)
                drink = rng.choice(list(DRINK_PRICE))
                distance = rng.randint(4, 12)
            else:
                pizza_count = rng.randint(1, 2)
                toppings = rng.randint(0, 1)
                drink = "None"
                distance = rng.randint(10, DRONE_RANGE_KM)
        else:
            pizza_count = rng.randint(1, 5)
            toppings = None
            drink = None
            distance = None
        pizza_count = min(pizza_count, remaining)
        if rng.random() < PEAK_SHARE[day_type]:
            peak_start, peak_end = PEAK_RANGE_BY_DAY[day_type]
            arrival = rng.randint(
                peak_start,
                peak_end - 1,
            )
        else:
            arrival = KITCHEN_START + rng.randint(0, KITCHEN_END - KITCHEN_START - 30)
        orders.append(
            order_row(
                order_id,
                arrival,
                rng,
                pizza_count=pizza_count,
                toppings=toppings,
                distance=distance,
                drink=drink,
            )
        )
        requested_pizzas += pizza_count
        order_id += 1
    fifo_orders = fifo_admission(orders)
    accepted_pizzas = sum(int(x["PizzaCount"]) for x in fifo_orders)
    return fifo_orders, requested - accepted_pizzas, orders


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


def sla_aware_ids(orders: list[dict[str, Any]]) -> list[int]:
    """Prioritize work that is most likely to create a downstream SLA delay.

    Within each five-minute arrival window, longer production jobs are started
    earlier so they do not accumulate at the back of the queue. Profit/ETA is
    used as a tie-breaker, preserving the business objective.
    """
    return [
        int(row["OrderID"])
        for row in sorted(
            orders,
            key=lambda x: (
                x["ArrivalMinute"] // 5,
                -x["EstimatedETA"],
                -priority_score(x),
                x["OrderID"],
            ),
        )
    ]


def optimizer_candidates(orders: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    """Generate transparent schedules for the local optimizer to evaluate."""
    return [
        ("FIFO", fifo_ids(orders)),
        ("Profit / ETA (5-minute window)", local_ai_ids(orders)),
        ("SLA-aware production sequencing", sla_aware_ids(orders)),
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


def labor_cost(tuesday: bool, extra_chefs: int = 0, extra_packers: int = 0) -> float:
    chefs = (CHEFS_TUESDAY if tuesday else CHEFS_NORMAL) + extra_chefs
    packers = PACKERS + extra_packers
    return (
        chefs * HOURLY_WAGES["Chef"] * SHIFT_HOURS
        + packers * HOURLY_WAGES["Packer"] * SHIFT_HOURS
        + DRONE_LOADERS * HOURLY_WAGES["Drone loader"] * SHIFT_HOURS
        + BATTERY_TECHNICIANS * HOURLY_WAGES["Battery technician"] * SHIFT_HOURS
    )


def simulate(
    orders: list[dict[str, Any]],
    sequence: list[int],
    tuesday: bool,
    extra_chefs: int = 0,
    extra_oven_chambers: int = 0,
    extra_packers: int = 0,
    surcharge_rate: float = 0.0,
) -> dict[str, Any]:
    by_id = {int(row["OrderID"]): row for row in orders}
    chefs = (CHEFS_TUESDAY if tuesday else CHEFS_NORMAL) + extra_chefs
    chef_available = [KITCHEN_START] * chefs
    packer_available = [KITCHEN_START] * (PACKERS + extra_packers)
    loader_available = [KITCHEN_START] * DRONE_LOADERS
    ovens = []
    for name, count in OVEN_CHAMBERS.items():
        for slot in range(count):
            ovens.append({"name": name, "available": KITCHEN_START, "c_slot": slot})
    for slot in range(extra_oven_chambers):
        ovens.append({"name": "Temporary Oven C", "available": KITCHEN_START, "c_slot": slot})
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
            warmup = OVEN_C_WARMUP if oven["name"] in {"Oven C", "Temporary Oven C"} else 0
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
        event_surcharge = order["Revenue"] * surcharge_rate
        charged_revenue = order["Revenue"] + drone_surcharge + event_surcharge
        refund, refund_label = compensation(elapsed, charged_revenue)
        oven_c_cost = OVEN_C_COST_PER_PIZZA * order["PizzaCount"] if oven["name"] in {"Oven C", "Temporary Oven C"} else 0
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
                "EventSurcharge": event_surcharge,
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
    event_surcharge_total = float(details_df["EventSurcharge"].sum()) if not details_df.empty else 0.0
    full_refunds = int((details_df["RefundOrCompensation"] == "Full refund").sum()) if not details_df.empty else 0
    labor = labor_cost(tuesday, extra_chefs, extra_packers)
    pre_tax = contribution - labor - DAILY_FIXED_COST
    adjusted_net = pre_tax * (1 - TAX_RATE) if pre_tax > 0 else pre_tax
    return {
        "details": details_df,
        "contribution": contribution,
        "total_compensation_cost": total_comp,
        "drone_surcharge_total": drone_surcharge_total,
        "event_surcharge_total": event_surcharge_total,
        "extra_chefs": extra_chefs,
        "extra_oven_chambers": extra_oven_chambers,
        "extra_packers": extra_packers,
        "surcharge_rate": surcharge_rate,
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


def run_evaluation(
    orders: list[dict[str, Any]],
    tuesday: bool,
    requested_orders: list[dict[str, Any]] | None = None,
    day_type: str = "Regular day",
) -> dict[str, Any]:
    admission_mode = requested_orders is not None and len(requested_orders) > len(orders)
    fifo_orders = fifo_admission(requested_orders) if admission_mode else orders
    ai_orders = ai_admission(requested_orders) if admission_mode else orders
    event_mode = day_type == "City event Thursday"
    policy_surcharge_rate = PEAK_SURCHARGE_RATE.get(day_type, 0.0)
    policy_extra_ovens = PEAK_EXTRA_OVEN_CHAMBERS.get(day_type, 0)
    policy_extra_packers = PEAK_EXTRA_PACKERS.get(day_type, 0)

    candidates = []
    fifo_candidates = optimizer_candidates(fifo_orders)
    # FIFO is the no-intervention baseline. PizzaFlow may activate the
    # approved event policy: one temporary chef and a transparent 10% demand
    # surcharge to protect capacity during the city event.
    fifo = simulate(fifo_orders, fifo_candidates[0][1], tuesday)
    for label, sequence in optimizer_candidates(ai_orders):
        candidates.append(
            (
                label,
                sequence,
                simulate(
                    ai_orders,
                    sequence,
                    tuesday,
                    extra_chefs=EVENT_EXTRA_CHEFS if event_mode else 0,
                    extra_oven_chambers=policy_extra_ovens,
                    extra_packers=policy_extra_packers,
                    surcharge_rate=policy_surcharge_rate,
                ),
            )
        )

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
    safety_fallback = selected_label == "FIFO" and not admission_mode
    profit_delta = ai["adjusted_net"] - fifo["adjusted_net"]
    profit_improvement = (profit_delta / fifo["adjusted_net"] * 100) if fifo["adjusted_net"] > 0 else 0
    eta_improvement = ((fifo["avg_eta"] - ai["avg_eta"]) / fifo["avg_eta"] * 100) if fifo["avg_eta"] else 0
    on_time_advantage = ai["on_time_rate"] - fifo["on_time_rate"]
    return {
        "fifo": fifo,
        "ai": ai,
        "profit_improvement": profit_improvement,
        "profit_delta": profit_delta,
        "eta_improvement": eta_improvement,
        "on_time_advantage": on_time_advantage,
        "sequence": ai_sequence,
        "safety_fallback": safety_fallback,
        "selected_label": selected_label,
        "fifo_orders": fifo_orders,
        "ai_orders": ai_orders,
        "admission_mode": admission_mode,
    }


def kpi_status(value: float, target: float, higher_is_better: bool = True) -> str:
    return "✅" if (value >= target if higher_is_better else value <= target) else "⚠️"


MANAGER_DECISIONS = [
    "Pending",
    "Approve recommendation",
    "Reject recommendation",
    "Request more information",
]


if "orders" not in st.session_state:
    st.session_state.orders = []
if "requested_orders" not in st.session_state:
    st.session_state.requested_orders = []
if "day_type" not in st.session_state:
    st.session_state.day_type = "Regular day"
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "tuesday" not in st.session_state:
    st.session_state.tuesday = False
if "assistant_answer" not in st.session_state:
    st.session_state.assistant_answer = ""
if "manager_decision" not in st.session_state:
    st.session_state.manager_decision = "Pending"
if "decision_history" not in st.session_state:
    st.session_state.decision_history = []


st.title("🍕 PizzaFlow AI")
st.caption("Explainable decision-support for a dark kitchen with autonomous drone delivery")
st.info("Local AI optimization engine + grounded manager assistant active. No API key, external service, or payment is required.")

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
            st.session_state.orders, rejected, st.session_state.requested_orders = build_orders(day_type, tuesday)
            st.session_state.day_type = day_type
            st.session_state.tuesday = tuesday
            st.session_state.evaluation = None
            st.session_state.assistant_answer = ""
            st.session_state.manager_decision = "Pending"
            accepted_pizzas = sum(int(x["PizzaCount"]) for x in st.session_state.orders)
            st.success(f"Loaded {accepted_pizzas} accepted pizzas in {len(st.session_state.orders)} orders. Rejected pizzas after capacity: {rejected}.")
    with c4:
        if st.button("🗑 Reset"):
            st.session_state.orders = []
            st.session_state.requested_orders = []
            st.session_state.evaluation = None
            st.session_state.assistant_answer = ""
            st.session_state.manager_decision = "Pending"
            st.session_state.decision_history = []
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
        display["EstimatedSLA_Buffer"] = SLA_MINUTES - display["EstimatedETA"]
        st.dataframe(display, use_container_width=True, height=360)
        if st.button("🤖 Analyze & Compare", type="primary"):
            st.session_state.evaluation = run_evaluation(
                st.session_state.orders,
                st.session_state.tuesday,
                st.session_state.requested_orders,
                st.session_state.day_type,
            )
            st.session_state.assistant_answer = ""
            st.session_state.manager_decision = "Pending"

        result = st.session_state.evaluation
        if result:
            fifo = result["fifo"]
            ai = result["ai"]
            stress_test = st.session_state.day_type == "City event Thursday"
            if stress_test:
                st.warning(
                    "Stress Test: 600 pizzas are requested, but the daily inventory cap accepts only 500. "
                    "The comparison is between FIFO's first 500 and PizzaFlow's value-ranked 500."
                )
            scenario_profit_target = {
                "Regular day": 0.0,
                "Weak Sunday": 0.0,
                "Busy Thursday": 10.0,
                "City event Thursday": 16.0,
            }[st.session_state.day_type]
            st.subheader("KPI Dashboard")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("AI on-time", f"{ai['on_time_rate']:.1f}%", delta_color="off")
            k1.caption(f"{kpi_status(ai['on_time_rate'], 98)} יעד: 98%")
            k2.metric(
                "Profit improvement",
                f"{result['profit_improvement']:.1f}%",
                delta_color="off",
            )
            k2.caption(
                f"{kpi_status(result['profit_improvement'], scenario_profit_target)} "
                f"יעד תרחיש: {scenario_profit_target:.0f}% | KPI מקורי: 15%"
            )
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

            if fifo["adjusted_net"] <= 0 and result["profit_delta"] > 0:
                st.success(
                    f"💰 PizzaFlow AI avoided a FIFO loss of {money(result['profit_delta'])}. "
                    "A percentage comparison is not meaningful when the FIFO baseline is negative."
                )

            st.subheader("FIFO vs PizzaFlow AI")
            if result["safety_fallback"]:
                st.info("The optimizer evaluated several schedules and selected FIFO because it was the best available schedule for this scenario.")
            else:
                st.success(f"The optimizer selected: {result['selected_label']}")
                if result["selected_label"] == "SLA-aware production sequencing":
                    st.info(
                        "The AI sequenced longer jobs earlier within each arrival window, "
                        "reducing queue buildup while keeping Profit/ETA as a tie-breaker."
                    )
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
                f"Peak surcharge: {money(ai['event_surcharge_total'])} | "
                f"Compensation/refunds: {money(ai['total_compensation_cost'])} | "
                f"Extra event chefs: {ai['extra_chefs']} | "
                f"Temporary oven chambers: {ai['extra_oven_chambers']} | "
                f"Extra packers: {ai['extra_packers']} | "
                f"Labor: {money(ai['labor'])} | Fixed rent/electricity: {money(ai['fixed'])} | "
                f"Tax rate: {TAX_RATE:.0%}"
            )
            st.subheader("Business Outcome")
            st.dataframe(ai["details"].head(30), use_container_width=True)

            st.subheader("👤 Human Approval and Decision Tracking")
            st.caption(
                "PizzaFlow is a decision-support system. The manager reviews the recommendation and records "
                "a decision before any operational change is made."
            )
            manager_decision = st.selectbox(
                "Manager decision",
                MANAGER_DECISIONS,
                key="manager_decision",
            )
            if st.button("💾 Save manager decision"):
                if manager_decision == "Pending":
                    st.warning("Choose Approve, Reject, or Request more information before saving.")
                else:
                    st.session_state.decision_history.append(
                        {
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "Scenario": st.session_state.day_type,
                            "Decision": manager_decision,
                            "AI policy": result["selected_label"],
                            "Profit improvement": f"{result['profit_improvement']:.1f}%",
                            "AI on-time": f"{ai['on_time_rate']:.1f}%",
                        }
                    )
                    st.success(f"Manager decision recorded: {manager_decision}")
            if st.session_state.decision_history:
                st.caption("Decision history")
                st.dataframe(
                    pd.DataFrame(st.session_state.decision_history),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("🧠 Local AI Manager Assistant")
            st.caption(
                "Ask questions in Hebrew or English. The assistant answers locally from the PizzaFlow "
                "specification and the measured scenario; it does not alter orders, KPIs, prices, or schedules."
            )
            manager_question = st.text_area(
                "Manager question",
                value="האם כדאי להפעיל את מדיניות השיא בתרחיש הזה, ומה הסיכון המרכזי?",
                key="manager_question",
            )
            if st.button("🧠 Ask the local assistant"):
                st.session_state.assistant_answer = local_manager_answer(
                    manager_question,
                    st.session_state.day_type,
                    result,
                )
            if st.session_state.assistant_answer:
                st.markdown(st.session_state.assistant_answer)
                st.caption("מקור התשובה: מפרט PizzaFlow ותוצאות הסימולציה המקומית; אין חיבור למודל חיצוני.")

with tab_method:
    st.header("Method & Responsible AI")
    st.markdown("""
**Objective:** Maximize Adjusted Net Profit subject to SLA ≤ 40 minutes, daily capacity of 500 pizzas, chef capacity, oven capacity, and drone capacity.

**Priority:** `Profit / ETA`, where ETA = preparation + baking + packaging + flight.

**Resources:** 3 chefs normally (2 on Tuesday), 1 packer, 1 drone loader, 1 battery technician, Oven A with 3 chambers, Oven B with 2, and Oven C with 1 chamber that needs 5 minutes warm-up and costs 10 ₪ per pizza. Under the approved option 2 peak policy, PizzaFlow can activate temporary peak capacity and a transparent peak surcharge; those additions are shown in the dashboard and included in net profit.

**Responsible AI:** The current deployment is a transparent local system. It uses only the supplied order fields, does not invent order IDs, and exposes its assumptions. A human remains responsible for operational decisions.

**Users and end-to-end process:** Operations staff create or load orders. The local AI engine calculates priority, tests candidate production sequences, evaluates the SLA and adjusted net profit, and presents the recommendation to the manager. The manager can approve the recommendation, reject it, or request more information. The selected decision is recorded in the dashboard so the scenario and decision can be reviewed later. Kitchen, packing and drone staff are the operational stakeholders who execute an approved plan.

**Tools and integration:** ChatGPT was used to help characterize the challenge and write the prototype code. Python, Streamlit and Plotly implement the working system, and GitHub plus Streamlit Community Cloud provide version control and deployment. The local assistant is grounded in the application specification and simulation output; no external connector or paid model is required.

**Local manager assistant:** The assistant uses local intent matching and grounded response templates over the project specification and the current simulation output. It requires no API key, sends no data outside the app, cannot change the simulation, and is not presented as a general-purpose generative language model. This makes the public prototype reproducible at zero service cost.
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
        "peak_surcharge_rate": PEAK_SURCHARGE_RATE,
        "peak_extra_oven_chambers": PEAK_EXTRA_OVEN_CHAMBERS,
        "peak_extra_packers": PEAK_EXTRA_PACKERS,
    })
