"""
Indonesian economic calendar events for realistic anomaly injection.

This module provides known economic events that caused price anomalies
in Indonesian food commodities. Events are sourced from:
- BI (Bank Indonesia) inflation reports
- BPS (Badan Pusat Statistik) price surveys
- Known policy changes (BBM, PPN)
- Major religious holidays (effect on supply/demand)

Each event has:
- date: when it occurred (or approximate)
- window: days before/after for the anomaly effect
- commodities: which products were affected
- effect: direction and approximate magnitude
- source: citation or reference
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class EconomicEvent:
    """An economic event that impacts food prices."""
    name: str
    start_date: date
    end_date: Optional[date] = None
    window_days: int = 7  # Default window ±days
    commodities: list[str] = field(default_factory=lambda: ["*"])  # "*" = all
    effect_type: str = "spike"  # spike, dip, structural_break
    magnitude: float = 0.15  # 15% price change
    source: str = ""
    confidence: str = "high"  # high, medium, low

    def date_range(self) -> list[date]:
        """Get all dates affected by this event."""
        if self.end_date:
            start = self.start_date
            end = self.end_date
        else:
            start = self.start_date - timedelta(days=self.window_days)
            end = self.start_date + timedelta(days=self.window_days)
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# === RELIGIOUS HOLIDAYS (recurring, high confidence) ===
# All major holidays affect food prices in Indonesia

HARI_RAYA = [
    # Idul Fitri / Lebaran — largest food price spike
    EconomicEvent("Idul Fitri", date(2020, 5, 24), window_days=14,
                  commodities=["*"], magnitude=0.30, effect_type="spike",
                  confidence="high",
                  source="BI: Ramadan inflation typically 0.5-1.5% month-on-month"),
    EconomicEvent("Idul Fitri", date(2021, 5, 13), window_days=14,
                  commodities=["*"], magnitude=0.25, effect_type="spike",
                  confidence="high"),
    EconomicEvent("Idul Fitri", date(2022, 5, 2), window_days=14,
                  commodities=["*"], magnitude=0.30, effect_type="spike",
                  confidence="high"),
    EconomicEvent("Idul Fitri", date(2023, 4, 22), window_days=14,
                  commodities=["*"], magnitude=0.25, effect_type="spike",
                  confidence="high"),
    EconomicEvent("Idul Fitri", date(2024, 4, 10), window_days=14,
                  commodities=["*"], magnitude=0.25, effect_type="spike",
                  confidence="high"),

    # Idul Adha — beef/lamb price spike
    EconomicEvent("Idul Adha", date(2020, 7, 31), window_days=7,
                  commodities=["daging sapi", "daging ayam", "daging"], magnitude=0.20,
                  effect_type="spike", confidence="high"),
    EconomicEvent("Idul Adha", date(2021, 7, 20), window_days=7,
                  commodities=["daging sapi", "daging ayam", "daging"], magnitude=0.20,
                  effect_type="spike", confidence="high"),
    EconomicEvent("Idul Adha", date(2022, 7, 10), window_days=7,
                  commodities=["daging sapi", "daging ayam", "daging"], magnitude=0.20,
                  effect_type="spike", confidence="high"),
    EconomicEvent("Idul Adha", date(2023, 6, 29), window_days=7,
                  commodities=["daging sapi", "daging ayam", "daging"], magnitude=0.20,
                  effect_type="spike", confidence="high"),
    EconomicEvent("Idul Adha", date(2024, 6, 17), window_days=7,
                  commodities=["daging sapi", "daging ayam", "daging"], magnitude=0.20,
                  effect_type="spike", confidence="high"),

    # Natal & Tahun Baru — poultry, eggs, luxury items
    EconomicEvent("Natal & Tahun Baru", date(2020, 12, 25), window_days=10,
                  commodities=["telur", "daging ayam", "minyak goreng"], magnitude=0.15,
                  effect_type="spike", confidence="medium"),
    EconomicEvent("Natal & Tahun Baru", date(2021, 12, 25), window_days=10,
                  commodities=["telur", "daging ayam", "minyak goreng"], magnitude=0.15,
                  effect_type="spike", confidence="medium"),
    EconomicEvent("Natal & Tahun Baru", date(2022, 12, 25), window_days=10,
                  commodities=["telur", "daging ayam", "minyak goreng"], magnitude=0.20,
                  effect_type="spike", confidence="medium"),
    EconomicEvent("Natal & Tahun Baru", date(2023, 12, 25), window_days=10,
                  commodities=["telur", "daging ayam", "minyak goreng"], magnitude=0.15,
                  effect_type="spike", confidence="medium"),

    # Nyepi — dip for some commodities
    EconomicEvent("Nyepi", date(2021, 3, 14), window_days=3,
                  commodities=["*"], magnitude=0.05, effect_type="dip",
                  confidence="low"),
    EconomicEvent("Nyepi", date(2022, 3, 3), window_days=3,
                  commodities=["*"], magnitude=0.05, effect_type="dip",
                  confidence="low"),
    EconomicEvent("Nyepi", date(2023, 3, 22), window_days=3,
                  commodities=["*"], magnitude=0.05, effect_type="dip",
                  confidence="low"),
    EconomicEvent("Nyepi", date(2024, 3, 11), window_days=3,
                  commodities=["*"], magnitude=0.05, effect_type="dip",
                  confidence="low"),
]


# === GOVERNMENT POLICY CHANGES (one-time, high confidence) ===

KEBIJAKAN = [
    # BBM price hike September 2022 — major structural break
    EconomicEvent("BBM Price Hike 2022", date(2022, 9, 3), window_days=30,
                  commodities=["*"], magnitude=0.35, effect_type="spike",
                  confidence="high",
                  source="Presidential announcement: Pertalite +30%, Solar +40%"),

    # BBM price hike 2014
    EconomicEvent("BBM Price Hike 2014", date(2014, 11, 18), window_days=30,
                  commodities=["*"], magnitude=0.25, effect_type="spike",
                  confidence="high",
                  source="President Jokowi: BBM subsidies removed, prices +30%"),

    # PPN increase 2022
    EconomicEvent("PPN 11% 2022", date(2022, 4, 1), window_days=14,
                  commodities=["*"], magnitude=0.05, effect_type="spike",
                  confidence="medium",
                  source="Government: PPN 10% → 11%"),

    # Minyak Goreng crisis 2022
    EconomicEvent("Minyak Goreng Crisis", date(2022, 3, 1), window_days=60,
                  commodities=["minyak goreng"], magnitude=0.50, effect_type="spike",
                  confidence="high",
                  source="Global CPO spike + DMO policy → cooking oil shortage"),
    EconomicEvent("Minyak Goreng Export Ban", date(2022, 4, 28), window_days=30,
                  commodities=["minyak goreng"], magnitude=0.30, effect_type="dip",
                  confidence="high",
                  source="President: CPO export ban → domestic price drop"),
]


# === NATURAL DISASTERS / SUPPLY SHOCKS ===

BENCANA = [
    # Jakarta floods
    EconomicEvent("Jakarta Flood 2020", date(2020, 1, 1), window_days=14,
                  commodities=["cabai", "sayuran", "bawang merah", "bawang putih"],
                  magnitude=0.20, effect_type="spike",
                  confidence="medium",
                  source="BMKG: Jakarta flood Jan 2020, supply chains disrupted"),

    # La Nina / extreme weather
    EconomicEvent("La Nina 2020-2021", date(2020, 10, 1),
                  end_date=date(2021, 4, 30), window_days=0,
                  commodities=["cabai", "bawang merah", "beras"],
                  magnitude=0.15, effect_type="spike",
                  confidence="medium",
                  source="BMKG: La Nina Oct 2020-Apr 2021, delayed harvest"),

    # El Nino 2023
    EconomicEvent("El Nino 2023", date(2023, 6, 1),
                  end_date=date(2024, 2, 29), window_days=0,
                  commodities=["beras", "cabai", "bawang merah"],
                  magnitude=0.25, effect_type="spike",
                  confidence="high",
                  source="BPS: El Nino caused rice production drop, price spike"),

    # Drought 2019
    EconomicEvent("Drought 2019", date(2019, 8, 1),
                  end_date=date(2019, 11, 30), window_days=0,
                  commodities=["beras", "cabai"],
                  magnitude=0.20, effect_type="spike",
                  confidence="medium",
                  source="BMKG: severe drought Aug-Nov 2019"),
]


# === COMBINED ===
ALL_EVENTS = HARI_RAYA + KEBIJAKAN + BENCANA


def get_events_for_date(target: date, commodity: str = "*") -> list[EconomicEvent]:
    """Get all events affecting a given date and commodity."""
    matching = []
    for event in ALL_EVENTS:
        if event.start_date <= target <= (event.end_date or event.start_date):
            if event.commodities[0] == "*" or commodity in event.commodities:
                matching.append(event)
    return matching


def get_event_calendar(start_year: int = 2020, end_year: int = 2025) -> dict[date, list[EconomicEvent]]:
    """Build a complete event calendar."""
    calendar = {}
    for event in ALL_EVENTS:
        for d in event.date_range():
            if d.year < start_year or d.year > end_year:
                continue
            if d not in calendar:
                calendar[d] = []
            calendar[d].append(event)
    return calendar


if __name__ == "__main__":
    calendar = get_event_calendar()
    print(f"Total event dates in calendar: {len(calendar)}")
    print(f"Total event definitions: {len(ALL_EVENTS)}")
    by_type = {}
    for event in ALL_EVENTS:
        by_type[event.confidence] = by_type.get(event.confidence, 0) + 1
    print(f"By confidence: {by_type}")
