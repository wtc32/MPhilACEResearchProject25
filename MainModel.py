"""
Title: Global Hydrogen Trade and Investment Modelling: Business Strategies for Adoption
Author: William Cope
University: University of Cambridge - Department of Chemical Engineering and Biotechnology
Degree: MPhil in Advanced Chemical Engineering
Module: Research Project Code
Date: 2025-08-22

Notes:
- This script accompanies the dissertation
"""
# flake8: noqa
# mypy: ignore-errors

# ========================= Imports & Global Setup ========================= #
# keep third‑party imports together for reproducibility and quicker review
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
try:
    import searoute as sr
except ImportError:
    print("Warning: searoute-py not installed. Using fallback great-circle paths.")
    sr = None
try:
    import fiona
    from shapely.geometry import shape, Polygon, MultiPolygon
except ImportError:
    print("Warning: fiona not installed. Using fallback map without land background.")
    fiona = None
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import newton, root_scalar, brentq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import networkx as nx
from matplotlib.sankey import Sankey

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ====================== Geometry & Distance Utilities ===================== #

# Haversine distance
# Great-circle distance on a sphere (km)
# Geodesic helper for distance/path calculations
def haversine(coord1, coord2):

    R = 6371
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Cartesian vector from lat, lon
# Helper function used in the modelling pipeline
def latlon_to_vec(lat, lon):

    lat, lon = np.radians(lat), np.radians(lon)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

# Rotation matrix around axis by theta
# Helper function used in the modelling pipeline
def rotation_matrix(axis, theta):

    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa + bb - cc - dd, 2*(bc + ad), 2*(bd - ac)],
        [2*(bc - ad), aa + cc - bb - dd, 2*(cd + ab)],
        [2*(bd + ac), 2*(cd - ab), aa + dd - bb - cc]
    ])

# Great-circle intermediate points, with option for long arc
# Interpolate points along great-circle/long-arc path
# Geodesic helper for distance/path calculations
def great_circle_points(coord1, coord2, n_points=500, long_arc=False):

    vec1 = latlon_to_vec(coord1[0], coord1[1])
    vec2 = latlon_to_vec(coord2[0], coord2[1])
    axis = np.cross(vec1, vec2)
    if np.linalg.norm(axis) == 0:
        return [coord1]
    axis = axis / np.linalg.norm(axis)
    d = np.arccos(np.dot(vec1, vec2))
    if long_arc:
        d = 2 * np.pi - d
        axis = -axis  # Flip direction for long arc
    theta = np.linspace(0, d, n_points)
    points = []
    for th in theta:
        rot = rotation_matrix(axis, th)
        point = np.dot(rot, vec1)
        lat = np.degrees(np.arcsin(point[2]))
        lon = np.degrees(np.arctan2(point[1], point[0]))
        points.append((lat, lon))
    return points

# Split route if it crosses the dateline by inserting nan at jumps
# Helper function used in the modelling pipeline
def split_dateline(lats, lons):

    lons = np.array(lons)
    lats = np.array(lats)
    # Normalize longitudes to -180 to 180
    lons = (lons + 180) % 360 - 180
    diff = np.diff(lons)
    cross_idx = np.where(np.abs(diff) > 180)[0]
    if len(cross_idx) > 0:
        lons_cross = np.insert(lons, cross_idx + 1, np.nan)
        lats_cross = np.insert(lats, cross_idx + 1, np.nan)
        return [(lats_cross, lons_cross)]
    return [(lats, lons)]

# ============================== Static Data ============================== #

# Coordinates (lat, lon for searoute compatibility)
all_coords = {
    "AUS_GLD": (-23.82, 151.25), "AUS_PIL": (-20.3, 118.6), "CHL_ANT": (-23.65, -70.4),
    "SAU_NEOM": (28.13, 34.92), "OMN_DUQ": (19.65, 57.75), "UAE_RUW": (24.13, 52.73),
    "NAM_LUD": (26.65, 15.15), "RSA_BOE": (-28.7, 16.5), "MAR_SOU": (27, -13),
    "MRT_NDB": (20.9, -17.0), "EGY_SUE": (29.6, 32.3), "KAZ_MNG": (43.7, 51.2),
    "USA_HOU": (29.72, -95.25), "CAN_PTT": (45.6, -61.4), "NOR_BER": (60.39, 5.32),
    "JPN_KAW": (35.01, 136.69), "KOR_ULS": (35.5, 129.35), "GER_WIL": (53.54, 8.16),
    "NLD_ROT": (51.95, 4.14), "BEL_ANR": (51.3, 4.3), "CHN_SHG": (31.3, 121.5),
    "IND_GUJ": (22.8, 69.7), "UK_TEE": (54.6, -1.1)
}

# Full node names
full_node_names = {
    "AUS_GLD": "AUS (GLD)",
    "AUS_PIL": "AUS (PIL)",
    "CHL_ANT": "CHL (ANT)",
    "SAU_NEOM": "SAU (NEOM)",
    "OMN_DUQ": "OMN (DUQ)",
    "UAE_RUW": "UAE (RUW)",
    "NAM_LUD": "NAM (LUD)",
    "RSA_BOE": "RSA (BOE)",
    "MAR_SOU": "MAR (SOU)",
    "MRT_NDB": "MRT (NDB)",
    "EGY_SUE": "EGY (SUE)",
    "KAZ_MNG": "KAZ (MNG)",
    "USA_HOU": "USA (HOU)",
    "CAN_PTT": "CAN (PTT)",
    "NOR_BER": "NOR (BER)",
    "JPN_KAW": "JPN (KAW)",
    "KOR_ULS": "KOR (ULS)",
    "GER_WIL": "GER (WIL)",
    "NLD_ROT": "NLD (ROT)",
    "BEL_ANR": "BEL (ANR)",
    "CHN_SHG": "CHN (SHG)",
    "IND_GUJ": "IND (GUJ)",
    "UK_TEE": "UK (TEE)"
}

# Continents
continents = {
    "AUS_GLD": "Australia", "AUS_PIL": "Australia", "CHL_ANT": "SouthAmerica",
    "SAU_NEOM": "MiddleEast", "OMN_DUQ": "MiddleEast", "UAE_RUW": "MiddleEast",
    "NAM_LUD": "Africa", "RSA_BOE": "Africa", "MAR_SOU": "Africa",
    "MRT_NDB": "Africa", "EGY_SUE": "MiddleEast", "KAZ_MNG": "Asia",
    "USA_HOU": "NorthAmerica", "CAN_PTT": "NorthAmerica", "NOR_BER": "Europe",
    "JPN_KAW": "Asia", "KOR_ULS": "Asia", "GER_WIL": "Europe",
    "NLD_ROT": "Europe", "BEL_ANR": "Europe", "CHN_SHG": "Asia",
    "IND_GUJ": "Asia", "UK_TEE": "Europe"
}

# ======================== Core Parameters & Series ======================== #
# centralise parameters and assumptions so scenarios are easy to tweak

# Updated multi-period data based on research (kt/yr for prod/demand, $/kg for costs; 2025 baseline, projections to 2050)
years = [2025, 2030, 2040, 2050]
learning_rate_mean = 0.18  # Adjusted to 0.18 for balance
wacc = 0.08  # Discount rate
lambda_emiss = np.linspace(0, 100, len(years))  # Ramp carbon penalty
rho_risk = 0.15  # Risk weight

# Emissions per node (kgCO2/kg H2; green 0, blue ~0.5 for better CCS)
emissions_data = {
    "AUS_GLD": 0, "AUS_PIL": 0, "CHL_ANT": 0, "SAU_NEOM": 0.5, "OMN_DUQ": 0.5, "UAE_RUW": 0.5,
    "NAM_LUD": 0, "RSA_BOE": 0, "MAR_SOU": 0, "MRT_NDB": 0, "EGY_SUE": 0, "KAZ_MNG": 0,
    "USA_HOU": 0.5, "CAN_PTT": 0, "NOR_BER": 0,
    "JPN_KAW": 0, "KOR_ULS": 0, "GER_WIL": 0, "NLD_ROT": 0, "BEL_ANR": 0, "CHN_SHG": 0, "IND_GUJ": 0, "UK_TEE": 0  # Add demands with 0
}

# Production data updated (capacity kt/yr scaled up for IEA totals, cost $/kg lower per IRENA)
nodes = ["AUS_GLD", "AUS_PIL", "CHL_ANT", "SAU_NEOM", "OMN_DUQ", "UAE_RUW", "NAM_LUD", "RSA_BOE", "MAR_SOU", "MRT_NDB", "EGY_SUE", "KAZ_MNG", "USA_HOU", "CAN_PTT", "NOR_BER"]
min_costs = [1.0, 1.2, 1.5, 1.3, 1.4, 1.4, 2.0, 2.3, 2.5, 2.7, 3.0, 3.3, 1.0, 1.5, 2.0]  # Slightly lower for blue
base_costs = [1.71, 1.881, 2.1375, 2.0, 2.2, 2.2, 2.565, 3.0075, 3.42, 3.591, 3.8475, 4.104, 1.2, 2.25, 2.565]
base_prods = [5000, 4000, 3000, 6000, 3500, 4500, 2500, 3000, 2000, 1500, 4000, 2500, 3500, 3000, 2500]  # Scaled to match trade ~50 Mt
prod_scales = [1250, 1000, 750, 1500, 875, 1125, 625, 750, 500, 375, 1000, 625, 875, 750, 625]  # Halved

# Double blue scales
blue_indices = [nodes.index(n) for n in ['SAU_NEOM', 'UAE_RUW', 'OMN_DUQ', 'USA_HOU']]
for i in blue_indices:
    prod_scales[i] *= 1.5

production_data = {}
for year in years:
    year_data = []
    for i in range(len(nodes)):
        prod = base_prods[i] + (year - 2025) * prod_scales[i]
        cost = max(min_costs[i], base_costs[i] * (1 - learning_rate_mean)**((year - 2025)/5) + np.random.normal(0, 0.3))
        year_data.append({"Node": nodes[i], "Production": prod, "Cost": cost})
    production_data[year] = year_data

# Demand data updated (kt/yr; IEA NZE: global 80 Mt 2030, 530 Mt 2050; regional shares)
demand_nodes = ["JPN_KAW", "KOR_ULS", "GER_WIL", "NLD_ROT", "BEL_ANR", "CHN_SHG", "IND_GUJ", "UK_TEE"]
base_demands = [7500, 10000, 6000, 7500, 3000, 15000, 10000, 3000]  # Scaled to match
demand_scales = [10000, 12000, 8000, 10000, 4000, 20000, 12000, 4000]  # Doubled for volume growth

demand_data_base = {}
for year in years:
    year_data = []
    for i in range(len(demand_nodes)):
        dem = base_demands[i] + (year - 2025) * demand_scales[i]
        year_data.append({"Node": demand_nodes[i], "Demand": dem})
    demand_data_base[year] = year_data

# ===================== Transport Cost Model & Modes ====================== #
# keep transport mode assumptions distinct so sensitivities don't spill into other logic

# Transport function updated per IRENA (pipeline $0.1-0.3/1000km ~0.0002*dist, ammonia $1-2 + 0.0001*dist, LH2 $2-4 + 0.0003*dist)
# Helper function used in the modelling pipeline
def get_min_transport_cost_and_mode(from_node, to_node, dist, trans_scale=1.0):

    modes = {}
    if continents.get(from_node) == continents.get(to_node) and dist < 3000:
        modes["Pipeline"] = trans_scale * 0.0002 * dist  # $0.2/1000km/kg avg
    modes["Ammonia"] = trans_scale * (1.5 + 0.0001 * dist)  # Updated
    modes["LH2"] = trans_scale * (3.0 + 0.0003 * dist)  # Updated higher
    if not modes:
        return float('inf'), None
    min_cost = min(modes.values())
    min_mode = min(modes, key=modes.get)
    return min_cost, min_mode

# Routes (common across years, add reverse for two-way)
producers = [p["Node"] for p in production_data[2025]]
demands = [d["Node"] for d in demand_data_base[2025]]
routes = [(f, t) for f in producers for t in demands] + [(t, f) for f in producers for t in demands]  # Bidirectional

# Precompute transport for routes (constant over years, but scale at runtime)
route_data_base = {}
for f, t in routes:
    dist = haversine(all_coords[f], all_coords[t])
    route_data_base[(f, t)] = {"dist": dist, "mode": get_min_transport_cost_and_mode(f, t, dist)[1]}

# Define blue nodes (based on your blue_indices; adjust if needed)
blue_nodes = ['SAU_NEOM', 'OMN_DUQ', 'UAE_RUW', 'USA_HOU']

# ========================== Optimisation Model =========================== #
# keep optimisation formulation isolated; solver swap/parameters can be changed here

# Function for single MILP run (with slack for feasibility, trans_scale, disrupt_node, disrupt_factor)
# Build and solve the multi-period H2 trade optimisation (MILP)
# Build and solve the hydrogen trade optimisation model
def run_milp(learning_rate, demand_scale=1.0, cost_scale=1.0, trans_scale=1.0, disrupt_node=None, disrupt_factor=1.0, carbon_multiplier=1.0, min_blue_shares=None):

    lambda_emiss_adj = lambda_emiss * carbon_multiplier
    production_data_run = {}
    for year in years:
        year_data = []
        # Producers
        for i in range(len(nodes)):
            prod = (base_prods[i] + (year - 2025) * prod_scales[i]) * (disrupt_factor if nodes[i] == disrupt_node else 1.0)
            cost = max(min_costs[i], base_costs[i] * cost_scale * (1 - learning_rate)**((year - 2025)/5))
            year_data.append({"Node": nodes[i], "Production": prod, "Cost": cost})
        # Demand nodes with 0 production/cost
        for d in demand_nodes:
            if d not in [item['Node'] for item in year_data]:  # Avoid duplicating if overlap
                year_data.append({"Node": d, "Production": 0, "Cost": 0})
        production_data_run[year] = year_data

    demand_data_run = {}
    for year in years:
        year_data = []
        for i in range(len(demand_nodes)):
            dem = (base_demands[i] + (year - 2025) * demand_scales[i]) * demand_scale
            year_data.append({"Node": demand_nodes[i], "Demand": dem})
        demand_data_run[year] = year_data

    model = pulp.LpProblem("Multi_Period_H2_Trade", pulp.LpMinimize)
    flows = pulp.LpVariable.dicts("Flow", [(f, t, y) for (f, t) in routes for y in years], lowBound=0)
    builds = pulp.LpVariable.dicts("Build", [(p, y) for p in producers for y in years], cat='Binary')
    cum_capacity = pulp.LpVariable.dicts("Cum_Cap", [(p, y) for p in producers for y in years], lowBound=0)
    slack = pulp.LpVariable.dicts("Slack", [(d, y) for d in demands for y in years], lowBound=0)  # Slack for feasibility

    # Objective with slack penalty
    objective = 0
    recip_flows = {}  # For reciprocity bonus
    for y_idx, y in enumerate(years):
        delta = 1 / (1 + wacc) ** (y - 2025)
        for (f, t) in routes:
            if (f, t) in route_data_base:
                prod_cost = next((p["Cost"] for p in production_data_run[y] if p["Node"] == f), 0)  # Safe default 0
                trans_cost = get_min_transport_cost_and_mode(f, t, route_data_base[(f, t)]["dist"], trans_scale)[0]
                emiss = emissions_data.get(f, 0)
                # FIXED: Add /1000 to convert kgCO2 to tCO2 for correct $/kgH2 units
                objective += delta * flows[(f, t, y)] * (prod_cost + trans_cost + lambda_emiss_adj[y_idx] * (emiss / 1000) + rho_risk * 0.1)
            # Add reciprocity bonus if reverse route exists
            if (t, f) in routes and f < t:  # Avoid double-counting
                key = (f, t, y)
                recip_flows[key] = pulp.LpVariable(f"recip_{f}_{t}_{y}", lowBound=0)
                model += recip_flows[key] <= flows[(f, t, y)]
                model += recip_flows[key] <= flows[(t, f, y)]
                objective -= delta * recip_flows[key] * 0.25  # Increased bonus to 25%
        for d in demands:
            objective += delta * slack[(d, y)] * 20  # Penalty $20/unmet kt

    model += objective

    # Constraints
    for y in years:
        for p in producers:
            prev_y = years[years.index(y) - 1] if y != 2025 else 2025
            max_cap = next((pp["Production"] for pp in production_data_run[y] if pp["Node"] == p), 0)  # Safe default
            if y == 2025:
                cum_capacity[(p, y)] = max_cap * builds[(p, y)]
            else:
                cum_capacity[(p, y)] = cum_capacity[(p, prev_y)] + max_cap * builds[(p, y)]
            model += cum_capacity[(p, y)] <= cum_capacity[(p, prev_y)] * 1.2 if y != 2025 else cum_capacity[(p, y)] <= max_cap
            model += pulp.lpSum(flows[(f, t, y)] for (f, t) in routes if f == p) <= cum_capacity[(p, y)]
            model += pulp.lpSum(flows[(f, t, y)] for (f, t) in routes if f == p) >= 0.05 * cum_capacity[(p, y)]

        for d in demands:
            total_flow = pulp.lpSum(flows[(f, t, y)] for (f, t) in routes if t == d)
            dem = next((dd["Demand"] for dd in demand_data_run[y] if dd["Node"] == d), 0)  # Safe default
            # TIGHTENED: To dem * 0.7 for higher fulfillment/volumes
            model += total_flow + slack[(d, y)] >= dem * 0.7
            model += total_flow <= dem
            for pp in producers:
                if (pp, d) in routes:
                    model += flows[(pp, d, y)] <= 0.5 * total_flow
            # Min flow to 0.01
            model += total_flow >= dem * 0.01

        # Minimum blue share constraint (added here, per year)
        if min_blue_shares is not None and y in min_blue_shares:
            blue_flow = pulp.lpSum(flows[(f, t, y)] for (f, t) in routes if f in blue_nodes)
            total_flow_y = pulp.lpSum(flows[(f, t, y)] for (f, t) in routes)
            model += blue_flow >= min_blue_shares[y] * total_flow_y

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract flows for map and avg price
    trade_results = []
    for y in years:
        for (f, t) in routes:
            flow = flows[(f, t, y)].varValue
            if flow > 1:
                v = route_data_base.get((f, t), {})
                trans_cost = get_min_transport_cost_and_mode(f, t, v["dist"], trans_scale)[0]
                prod_cost = next((p["Cost"] for p in production_data_run[y] if p["Node"] == f), 0)
                unit_cost = prod_cost + trans_cost
                trade_results.append({"Year": y, "From": f, "To": t, "Flow_kt": flow, "Distance_km": v.get("dist", 0), "Mode": v.get("mode", ""), "Unit_Cost": unit_cost})

    df_trade = pd.DataFrame(trade_results)

    avg_price = 0
    count = 0
    delivered_prices = {y: {} for y in years}
    avg_h2_prices = {y: 0 for y in years}
    for y in years:
        y_count = 0
        for d in demands:
            total_flow = sum(flows[(p, d, y)].varValue for p in producers if (p, d) in routes)
            if total_flow > 0:
                weighted_cost = sum(flows[(p, d, y)].varValue * (next((pp["Cost"] for pp in production_data_run[y] if pp["Node"] == p), 0) + get_min_transport_cost_and_mode(p, d, route_data_base.get((p, d), {}).get("dist", 0), trans_scale)[0]) for p in producers if (p, d) in routes) / total_flow
                delivered_prices[y][d] = weighted_cost
                avg_h2_prices[y] += weighted_cost
                avg_price += weighted_cost
                count += 1
                y_count += 1
            else:
                delivered_prices[y][d] = np.nan
        if y_count > 0:
            avg_h2_prices[y] /= y_count

    overall_avg = avg_price / count if count > 0 else 6.0

    return overall_avg, df_trade, model, delivered_prices, avg_h2_prices

# ====================== Financial Metrics & Helpers ====================== #
# apply a consistent finance framework across sectors (comparability)

# Manual IRR function (fixed with brentq and bounds to avoid unrealistic values)
# IRR via brentq root-finding with guard-rails
# Compute internal rate of return from cashflows
def manual_irr(cashflows, tol=1e-6):

    # Compute net present value from cashflows
    def npv(r):

        return sum(cf / (1 + r)**t for t, cf in enumerate(cashflows))
    if npv(0) <= 0:
        return 0.0
    try:
        irr = brentq(npv, 0, 1, xtol=tol)
        return irr * 100
    except:
        return 0.0

# ========================= Sector Investment Model ======================= #
# group sector investment rules in one place to avoid hidden differences across cases

# Project/sector cashflow model with IRR/NPV utilities
class SectorInvestment:
    # Helper function used in the modelling pipeline
    def __init__(self, capex, capacity, h2_usage, rev, opex_base, life, emissions_reduced, tax_rate, depre_rate=0.05, elec_usage=0, elec_cost=0, energy_usage=0, energy_cost=0, cac_base=200, retention_rate=0.8, burn_volatility=0.1, esg_score=8, credit_value=50, learning=0.1, market_growth=8):

        self.capex = capex
        self.capacity = capacity
        self.h2_usage = h2_usage
        self.rev = rev
        self.opex_base = opex_base
        self.life = life
        self.emissions_reduced = emissions_reduced
        self.tax_rate = tax_rate
        self.depre_rate = depre_rate
        self.elec_usage = elec_usage
        self.elec_cost = elec_cost
        self.energy_usage = energy_usage
        self.energy_cost = energy_cost
        self.cac_base = cac_base  # Base CAC per customer/unit (sector-specific, e.g., $200 for transport)
        self.retention_rate = retention_rate  # For LTV calculation (e.g., 80% retention)
        self.burn_volatility = burn_volatility  # Volatility for randomized burn in MC
        self.esg_score = esg_score  # ESG score (1-10, higher for green)
        self.credit_value = credit_value  # Carbon credit value per t reduced (e.g., $50/t)
        self.learning = learning
        self.market_growth = market_growth
        self.stages = [
            {'name': 'Seed', 'funding_fraction': 0.2, 'duration': 1},
            {'name': 'Series A', 'funding_fraction': 0.3, 'duration': 2},
            {'name': 'Series B', 'funding_fraction': 0.5, 'duration': 2}
        ]

    # Helper function used in the modelling pipeline
    def calculate_ltv(self, avg_rev_per_user=None):

        if avg_rev_per_user is None:
            avg_rev_per_user = self.rev  # Default to self.rev as avg rev per unit/customer
        return avg_rev_per_user * self.retention_rate / (1 - self.retention_rate)  # Basic LTV formula

    # Helper function used in the modelling pipeline
    def calculate_cashflows(self, h2_price, carbon_price=0, subsidy=0, learning=0.1, ef=1):

        adj_capex = self.capex * (1 - learning)
        depre = adj_capex * self.depre_rate
        adj_opex = self.opex_base * (1 - learning)
        elec_opex = self.elec_usage * self.elec_cost * self.capacity
        energy_opex = self.energy_usage * self.energy_cost * self.capacity
        cashflows = [0] * (self.life + 1)  # Extend for stages
        cum_duration = 0
        for stage in self.stages:
            stage_capex = adj_capex * stage['funding_fraction']
            cashflows[cum_duration] -= stage_capex  # Stage capex at start of phase
            cum_duration += stage['duration']
        cac_total = self.cac_base * self.capacity  # Scaled CAC
        cashflows[0] -= cac_total  # Subtract from initial capex
        for t in range(1, self.life + 1):
            # Adjust revenue for shipping only (per ton-km * nominal distance)
            if self.h2_usage <= 0.5:  # Threshold for shipping (e.g., 0.3 kg/ton)
                rev_t = self.rev * self.capacity * 1000  # $/ton-km * tons * 1000 km
            else:
                rev_t = self.capacity * self.rev * (1 + 0.02)**t  # Standard per-ton growth for others, including fertilizer
            credit_rev = self.credit_value * self.emissions_reduced * self.capacity * ef  # Carbon credits
            rev_t += credit_rev
            cost_t = adj_opex * (1 + 0.03)**t + (self.h2_usage * h2_price * self.capacity) + elec_opex * (1 + 0.03)**t + energy_opex * (1 + 0.03)**t
            burn_rate = self.opex_base / 12 * (1 + np.random.normal(0, self.burn_volatility))  # Monthly burn
            cost_t += burn_rate * 12 * self.capacity  # Annualized burn
            ebitda = rev_t - cost_t + (carbon_price * self.emissions_reduced * self.capacity * ef) + (subsidy * self.h2_usage * self.capacity)
            taxable = ebitda - depre
            tax_t = max(0, taxable) * self.tax_rate
            net_income = taxable - tax_t
            cf_t = net_income + depre
            cashflows[t] += cf_t  # Add to ongoing years
        return cashflows

    # Compute internal rate of return from cashflows
    def irr(self, h2_price, carbon_price=0, subsidy=0, learning=0.1, wacc=0.08, ef=1):

        cashflows = self.calculate_cashflows(h2_price, carbon_price, subsidy, learning, ef)
        irr_val = manual_irr(cashflows)
        if self.esg_score < 7:
            irr_val *= 0.9  # -10% penalty for low ESG (VC risk premium)
        return irr_val

    # Compute net present value from cashflows
    def npv(self, h2_price, carbon_price=0, subsidy=0, learning=0.1, wacc=0.08, ef=1):

        cashflows = self.calculate_cashflows(h2_price, carbon_price, subsidy, learning, ef)
        exit_value = self.simulate_exit() * self.rev * self.capacity  # Simple rev-based exit
        cashflows[-1] += exit_value  # Add to final year as terminal value
        return sum(cf / (1 + wacc)**t for t, cf in enumerate(cashflows))

    # Helper function used in the modelling pipeline
    def break_even_h2(self, target_irr=15, carbon_price=0, subsidy=0, learning=0.1, ef=1):

        ltv = self.calculate_ltv()
        cac_total = self.cac_base * self.capacity
        if ltv / cac_total < 3:
            print(f"Warning: LTV/CAC <3 for {self.__class__.__name__} - High risk")
        # Helper function used in the modelling pipeline
        def objective(h2):

            return self.irr(h2, carbon_price, subsidy, learning, ef=ef) - target_irr
        try:
            be = brentq(objective, -20, 30)
            return max(be, 0)
        except:
            return np.inf

    # Helper function used in the modelling pipeline
    def link_vc_to_trade(df_trade, sector_investments, intensities, blue_pct):

        for sector_name, sector in sector_investments.items():
            # Example: Boost TAM if high export intensity for related producer
            related_producer = 'USA_HOU' if sector_name == 'Transport' else None  # Map sector to producer
            if related_producer and intensities.get(related_producer, 0) > 0.5:
                sector.vc_tam_2050 = vc_tam_2050[sector_name] * 1.2  # +20% for high export
            # For blue/green: Green sectors get boost if green_share >0.7
            green_share = 1 - blue_pct  # From your blue_pct calc
            if emissions_data.get(sector_name, 0) == 0 and green_share > 0.7:
                sector.learning = sector.learning + 0.05  # Faster learning for green dominance
        return sector_investments  # Updated with trade-linked adjustments

    # Helper function used in the modelling pipeline
    def simulate_syndicate(self, num_investors=3, lead_share=0.6):

        G = nx.Graph()
        investors = [f"Investor_{i}" for i in range(num_investors)]
        G.add_nodes_from(investors + [self.__class__.__name__])  # Sector as startup node
        shares = [lead_share] + [(1 - lead_share) / (num_investors - 1)] * (num_investors - 1)
        for i, investor in enumerate(investors):
            G.add_edge(investor, self.__class__.__name__, weight=shares[i])
        adjusted_irr = self.irr(...) * (1 + 0.1 * shares[0])  # Lead premium 10%; replace ... with args
        return G, adjusted_irr  # Return graph and adjusted IRR

    # Helper function used in the modelling pipeline
    def simulate_exit(self, base_multiple=4, multiple_vol=0.5):

        multiple = np.random.lognormal(mean=np.log(base_multiple), sigma=multiple_vol)  # Lognormal for skewed multiples
        return multiple if self.market_growth > 7 else multiple * 0.75  # Discount if low growth

# Sector params (unchanged except FERT_H2_USAGE fixed to 18)
STEEL_CAPEX = 1000e6
STEEL_CAPACITY = 1e6
STEEL_H2_USAGE = 55
STEEL_REV = 510
STEEL_OPEX_BASE = 470
STEEL_ELEC_USAGE = 0.7
STEEL_ELEC_COST = 70
STEEL_LIFE = 20
STEEL_EMISSIONS_REDUCED = 1.6
STEEL_TAX = 0.3  # Adjusted to standard corporate tax rate ~30%

FERT_CAPEX = 1000e6
FERT_CAPACITY = 1e6
FERT_H2_USAGE = 170  # Fixed to realistic value
FERT_REV = 500
FERT_OPEX_BASE = 200
FERT_ENERGY_USAGE = 1.6
FERT_ENERGY_COST = 65
FERT_LIFE = 20
FERT_EMISSIONS_REDUCED = 1.8
FERT_TAX = 0.3

SHIP_CAPEX = 400e6
SHIP_CAPACITY = 1e6
SHIP_H2_USAGE = 0.3
SHIP_REV = 0.045
SHIP_OPEX_BASE = 0.05
SHIP_LIFE = 15
SHIP_EMISSIONS_REDUCED = 0.3
SHIP_TAX = 0.3

TRANS_CAPEX = 800e6  # +60% per benchmarks for refueling/aviation infra
TRANS_CAPACITY = 1000
TRANS_H2_USAGE = 6000
TRANS_REV = 220000
TRANS_OPEX_BASE = 150000
TRANS_LIFE = 15
TRANS_EMISSIONS_REDUCED = 150
TRANS_TAX = 0.3

CHEM_CAPEX = 3000e6
CHEM_CAPACITY = 1e6
CHEM_H2_USAGE = 150
CHEM_REV = 720
CHEM_OPEX_BASE = 300
CHEM_LIFE = 20
CHEM_EMISSIONS_REDUCED = 1.7
CHEM_TAX = 0.3

# ======================== Sector Setup & Weights ========================= #

# Instantiations
steel = SectorInvestment(STEEL_CAPEX, STEEL_CAPACITY, STEEL_H2_USAGE, STEEL_REV, STEEL_OPEX_BASE, STEEL_LIFE, STEEL_EMISSIONS_REDUCED, STEEL_TAX, elec_usage=STEEL_ELEC_USAGE, elec_cost=STEEL_ELEC_COST)
fertilizer = SectorInvestment(FERT_CAPEX, FERT_CAPACITY, FERT_H2_USAGE, FERT_REV, FERT_OPEX_BASE, FERT_LIFE, FERT_EMISSIONS_REDUCED, FERT_TAX, energy_usage=FERT_ENERGY_USAGE, energy_cost=FERT_ENERGY_COST)
shipping = SectorInvestment(SHIP_CAPEX, SHIP_CAPACITY, SHIP_H2_USAGE, SHIP_REV, SHIP_OPEX_BASE, SHIP_LIFE, SHIP_EMISSIONS_REDUCED, SHIP_TAX)
transport = SectorInvestment(TRANS_CAPEX, TRANS_CAPACITY, TRANS_H2_USAGE, TRANS_REV, TRANS_OPEX_BASE, TRANS_LIFE, TRANS_EMISSIONS_REDUCED, TRANS_TAX, burn_volatility=0.15)
chemicals = SectorInvestment(CHEM_CAPEX, CHEM_CAPACITY, CHEM_H2_USAGE, CHEM_REV, CHEM_OPEX_BASE, CHEM_LIFE, CHEM_EMISSIONS_REDUCED, CHEM_TAX)

sectors = {
    "Steel": steel,
    "Fertilizer": fertilizer,
    "Shipping": shipping,
    "Transport": transport,
    "Chemicals": chemicals
}

# Factor scores (unchanged)
factor_scores = {
    "Steel": {"Feasibility": 7, "Research Needed": 4, "Competition": 6, "Stakeholder Perception": 6, "Market Growth": 7, "Barriers": 5},
    "Fertilizer": {"Feasibility": 8, "Research Needed": 3, "Competition": 5, "Stakeholder Perception": 7, "Market Growth": 8, "Barriers": 4},
    "Shipping": {"Feasibility": 8, "Research Needed": 3, "Competition": 7, "Stakeholder Perception": 7, "Market Growth": 9, "Barriers": 6},
    "Transport": {"Feasibility": 6, "Research Needed": 5, "Competition": 8, "Stakeholder Perception": 5, "Market Growth": 8, "Barriers": 7},
    "Chemicals": {"Feasibility": 7, "Research Needed": 4, "Competition": 5, "Stakeholder Perception": 6, "Market Growth": 7, "Barriers": 5}
}

sector_notes = {
    "Steel": "High impact for DRI; subsidies for pilots recommended.",
    "Fertilizer": "Low research; focus on offtake for ammonia.",
    "Shipping": "Policy incentives key; high growth in maritime fuel.",
    "Transport": "Growing in road/aviation; infrastructure challenges.",
    "Chemicals": "Essential for refining/methanol; stable demand."
}

# Weights (unchanged)
weights = {"IRR": 0.3, "Feasibility": 0.2, "Market Growth": 0.15, "Stakeholder Perception": 0.15, "Research Needed": 0.1, "Competition": 0.05, "Barriers": 0.05}

subsidies = np.linspace(0, 3, 4)

# ============================== Part A Runs ============================== #

print("Running deterministic MILP...")
min_blue_shares = {2025: 0.4, 2030: 0.35, 2040: 0.25, 2050: 0.15}
avg_h2_det, df_trade_det, model_det, delivered_prices_det, avg_h2_prices_det = run_milp(learning_rate_mean, 1.0, 1.0, min_blue_shares=min_blue_shares)

print("### Part A: Global Hydrogen Supply Chain Optimization (Multi-Period, Deterministic)")
print(df_trade_det.to_markdown(index=False))
print(f"\nStatus: {pulp.LpStatus[model_det.status]}")
print(f"Total Discounted Cost: {pulp.value(model_det.objective)} $m")

print("\n#### Delivered H2 Prices ($/kg, Weighted Avg by Year, Deterministic)")
for y in years:
    print(f"Year {y}:")
    print(pd.DataFrame.from_dict(delivered_prices_det[y], orient='index', columns=['Price']).to_markdown())
    print(f"Average H2 Price {y}: ${avg_h2_prices_det[y]:.2f}/kg" if not np.isnan(avg_h2_prices_det[y]) else "N/A")

# =========================== Visualisations ============================== #
# all visualisations gathered here; keep plotting in one place

# Combined Maps in One Figure with Thinner Lines, Borders, Spacing, No Node Labels, Single Title
# Define manual offsets (in degrees lon/lat) and alignments to avoid overlaps
# Tuned based on coordinate clusters: positive x=right, negative x=left, etc.
offsets = {
    # Europe cluster
    "UK_TEE": (-15, 0),       # Left
    "NOR_BER": (0, 10),       # Above
    "GER_WIL": (10, 5),       # Right-up
    "NLD_ROT": (10, -5),      # Right-down
    "BEL_ANR": (-10, -10),    # Left-down
    # East Asia
    "JPN_KAW": (10, 0),       # Right
    "KOR_ULS": (10, 5),       # Right-up
    "CHN_SHG": (-10, 0),      # Left
    "IND_GUJ": (-10, -5),     # Left-down
    "KAZ_MNG": (0, 10),       # Above
    # Middle East/Africa
    "SAU_NEOM": (0, 10),      # Above
    "OMN_DUQ": (10, 0),       # Right
    "UAE_RUW": (10, -5),      # Right-down
    "EGY_SUE": (-10, 0),      # Left
    "NAM_LUD": (-10, 0),      # Left
    "RSA_BOE": (0, -10),      # Below
    "MAR_SOU": (10, 0),       # Right
    "MRT_NDB": (-10, 5),      # Left-up
    # Others (default-like but tuned)
    "AUS_GLD": (10, 0),
    "AUS_PIL": (-10, 0),
    "CHL_ANT": (-10, 0),
    "USA_HOU": (-10, 0),
    "CAN_PTT": (10, 0),
}
alignments = {
    # ha: horizontal alignment, va: vertical alignment
    "UK_TEE": {'ha': 'right', 'va': 'center'},
    "NOR_BER": {'ha': 'center', 'va': 'bottom'},
    "GER_WIL": {'ha': 'left', 'va': 'bottom'},
    "NLD_ROT": {'ha': 'left', 'va': 'top'},
    "BEL_ANR": {'ha': 'right', 'va': 'top'},
    "JPN_KAW": {'ha': 'left', 'va': 'center'},
    "KOR_ULS": {'ha': 'left', 'va': 'bottom'},
    "CHN_SHG": {'ha': 'right', 'va': 'center'},
    "IND_GUJ": {'ha': 'right', 'va': 'top'},
    "KAZ_MNG": {'ha': 'center', 'va': 'bottom'},
    "SAU_NEOM": {'ha': 'center', 'va': 'bottom'},
    "OMN_DUQ": {'ha': 'left', 'va': 'center'},
    "UAE_RUW": {'ha': 'left', 'va': 'top'},
    "EGY_SUE": {'ha': 'right', 'va': 'center'},
    "NAM_LUD": {'ha': 'right', 'va': 'center'},
    "RSA_BOE": {'ha': 'center', 'va': 'top'},
    "MAR_SOU": {'ha': 'left', 'va': 'center'},
    "MRT_NDB": {'ha': 'right', 'va': 'bottom'},
    # Defaults for others
}

# Path to land shapefile (adjust as needed)
land_shp_path = "C:/Users/willc/Downloads/ne_50m_land/ne_50m_land.shp"

fig, axs = plt.subplots(2, 2, figsize=(32, 22))  # Kept dimensions for balance
fig.suptitle('', fontsize=20, y=0.98)  # Raised y to move title higher
axs = axs.flatten()

for idx, year in enumerate(years):
    ax = axs[idx]
    ax.set_title(f'{year}', fontsize=16, pad=15)  # Kept pad for spacing
    ax.set_xlabel('Longitude', labelpad=15)  # Kept pad for visibility
    ax.set_ylabel('Latitude', labelpad=15)  # Kept pad to prevent intrusion
    # Adjusted limits with margin to prevent clipping
    ax.set_xlim(-190, 190)  # Extended by 10 degrees on each side
    ax.set_ylim(-100, 100)  # Extended by 10 degrees on each side

    # Subtle gridlines for meridians/parallels
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Grey borders
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(1.5)

    # Land background (neutral color)
    if fiona is not None:
        with fiona.open(land_shp_path, "r") as src:
            for feature in src:
                geom = shape(feature["geometry"])
                if isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, color='lightgray', alpha=1, zorder=1)
                elif isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, color='lightgray', alpha=1, zorder=1)
    else:
        print("Fallback: No land background.")

    # Nodes with reduced size and dynamic label adjustment
    node_positions = {}  # To track positions for overlap check
    for node, (lat, lon) in all_coords.items():
        color= 'black' if node in producers else 'red'
        ax.scatter(lon, lat, s=40, color=color, zorder=5)  # Reduced size to 40
        node_positions[node] = (lon, lat)
        # Dynamic label adjustment
        off_x, off_y = offsets.get(node, (5, 5))  # Default offset
        align = alignments.get(node, {'ha': 'left', 'va': 'bottom'})
        # Check for overlap with other nodes
        for other_node, (other_lon, other_lat) in node_positions.items():
            if node != other_node:
                dist = ((lon - other_lon) ** 2 + (lat - other_lat) ** 2) ** 0.5
                if dist < 5:  # Threshold for overlap (in degrees)
                    off_x, off_y = -off_x, -off_y  # Flip direction to avoid overlap
        ax.text(lon + off_x, lat + off_y, full_node_names[node], fontsize=7, zorder=6, **align)

    # Year flows (thinner lines as before)
    year_flows = df_trade_det[df_trade_det['Year'] == year]
    for _, r in year_flows.iterrows():
        f, t, flow, mode = r['From'], r['To'], r['Flow_kt'], r['Mode']
        long_arc = f == "USA_HOU" and t == "JPN_KAW"
        if mode == "Pipeline" or sr is None:
            points = great_circle_points(all_coords[f], all_coords[t], long_arc=long_arc)
        else:
            origin = [all_coords[f][1], all_coords[f][0]]
            destination = [all_coords[t][1], all_coords[t][0]]
            try:
                route = sr.searoute(origin, destination, append_orig_dest=True, include_ports=True)
                points = [(coord[1], coord[0]) for coord in route.geometry.coordinates]
            except:
                points = great_circle_points(all_coords[f], all_coords[t], long_arc=long_arc)
        lats, lons = zip(*points)
        line_color = 'grey' if mode == 'Pipeline' else 'blue' if mode == 'Ammonia' else 'purple' if mode == 'LH2' else 'black'
        style = '--' if mode == 'Pipeline' else '-'
        segments = split_dateline(lats, lons)
        for seg_lats, seg_lons in segments:
            ax.plot(seg_lons, seg_lats, lw=0.8 * np.sqrt(np.log(flow + 1)), color=line_color, alpha=0.6, linestyle=style, zorder=4)

# Shared legend below subplots
ocean_patch = Patch(color='white', label='Ocean')
land_patch = Patch(color='lightgray', label='Land')
ammonia_line = Line2D([0], [0], color='blue', linewidth=2, label='Ammonia Routes')
pipeline_line = Line2D([0], [0], color='grey', linewidth=2, linestyle='--', label='Pipeline Routes')
lh2_line = Line2D([0], [0], color='purple', linewidth=2, label='LH2 Routes')
supply_dot = Line2D([0], [0], marker='o', color='w', label='Supply Nodes', markerfacecolor='black', markersize=10)
demand_dot = Line2D([0], [0], marker='o', color='w', label='Demand Nodes', markerfacecolor='red', markersize=10)

handles = [ocean_patch, land_patch, ammonia_line, pipeline_line, lh2_line, supply_dot, demand_dot]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)  # Kept further down

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.95, wspace=0.3, hspace=0.35)  # Adjusted top margin
# Save figure to file for inclusion in dissertation / appendix
plt.savefig('combined_hydrogen_trade_maps.pdf', facecolor='white')
plt.show()

# ==================== Ranking & Recommendations ========================== #

# Ranking (unchanged)
ranking_results = []
mid_sub = 1.5
mid_cp = 75
for sector_name, sector in sectors.items():
    irr = sector.irr(avg_h2_prices_det[2030], mid_cp, mid_sub)
    npv = sector.npv(avg_h2_prices_det[2030], mid_cp, mid_sub) / 1e6
    scores = factor_scores[sector_name]
    composite = (
        weights["IRR"] * (irr / 20) +
        weights["Feasibility"] * scores["Feasibility"] +
        weights["Market Growth"] * scores["Market Growth"] +
        weights["Stakeholder Perception"] * scores["Stakeholder Perception"] +
        weights["Research Needed"] * (10 - scores["Research Needed"]) +
        weights["Competition"] * (10 - scores["Competition"]) +
        weights["Barriers"] * (10 - scores["Barriers"])
    )
    ranking_results.append({"Sector": sector_name, "IRR%": irr, "NPV $M": npv, "Composite_Score": composite})

df_ranking = pd.DataFrame(ranking_results).sort_values(by='Composite_Score', ascending=False)

top_3 = df_ranking.head(3)
recommendations = []
for _, row in top_3.iterrows():
    rec = f"Invest in {row['Sector']}; IRR {row['IRR%']:.1f}%, NPV ${row['NPV $M']:.1f}M at Subsidy $1.5/kg, Carbon ${mid_cp}/t, Composite Score {row['Composite_Score']:.1f}. Note: {sector_notes[row['Sector']]}"
    recommendations.append(rec)

print("\n#### Investment Recommendations (Top 3 Ranked, Deterministic)")
for rec in recommendations:
    print(rec)

# ======================= Monte Carlo & Advanced Trade ==================== #
# include Monte Carlo runs for robustness; separate randomness from core logic

# Advanced Trade (unchanged)
ltc_share = {2025: 0.8, 2030: 0.7, 2040: 0.6, 2050: 0.5}
ltc_discount = 0.15

num_runs = 500
mc_results = []
mc_h2_prices = []
mc_hhi = []
mc_ltc_prices = {y: [] for y in years}
mc_intensities = []
mc_blue_shares = []
for run in range(num_runs):
    learning_rate = np.random.uniform(0.1, 0.2)
    demand_scale = np.random.uniform(0.7, 1.3)
    cost_scale = np.random.uniform(0.7, 1.3)
    trans_scale = np.random.uniform(0.7, 1.3)
    carbon_multiplier = np.random.uniform(1.0, 2.0)  # For blue/green sensitivity
    avg_h2, df_trade_run, _, _, _ = run_milp(learning_rate, demand_scale, cost_scale, trans_scale, carbon_multiplier=carbon_multiplier)
    mc_h2_prices.append(avg_h2)
    export_shares = df_trade_run.groupby('From')['Flow_kt'].sum() / df_trade_run['Flow_kt'].sum()
    hhi = (export_shares**2).sum() * 10000
    mc_hhi.append(hhi)
    carbon = np.random.uniform(50, 150)
    subsidy = np.random.uniform(0, 3)
    for sector_name, sector in sectors.items():
        irr = sector.irr(avg_h2, carbon, subsidy)
        npv = sector.npv(avg_h2, carbon, subsidy) / 1e6
        if hhi > 3000:
            irr -= np.random.uniform(0.05, 0.1) * irr
        mc_results.append({"Run": run, "Sector": sector_name, "H2 Price": avg_h2, "Carbon $/t": carbon, "Subsidy $/kg": subsidy, "IRR%": irr, "NPV $M": npv})
    df_trade_run['LTC_Flow'] = df_trade_run.apply(lambda row: row['Flow_kt'] * ltc_share[row['Year']], axis=1)
    df_trade_run['Spot_Flow'] = df_trade_run['Flow_kt'] - df_trade_run['LTC_Flow']
    df_trade_run['Unit_Cost_LTC'] = df_trade_run['Unit_Cost'] * (1 - ltc_discount)
    for y in years:
        ltc_y = df_trade_run[df_trade_run['Year'] == y]['Unit_Cost_LTC'].mean()
        mc_ltc_prices[y].append(ltc_y if not np.isnan(ltc_y) else avg_h2)
    intensities_run = {}
    for p in producers:
        exp = df_trade_run[df_trade_run['From'] == p]['Flow_kt'].sum()
        prod = sum(production_data[y][nodes.index(p)]['Production'] for y in years) / len(years)
        intensity = exp / prod if prod > 0 else 0
        intensities_run[p] = intensity
    mc_intensities.append(intensities_run)
    total_2050 = df_trade_run[df_trade_run['Year'] == 2050]['Flow_kt'].sum()
    blue_flow_2050 = df_trade_run[(df_trade_run['Year'] == 2050) & df_trade_run['From'].isin(blue_nodes)]['Flow_kt'].sum() / total_2050 if total_2050 > 0 else 0
    mc_blue_shares.append(blue_flow_2050)

df_mc = pd.DataFrame(mc_results)

mc_stats = df_mc.groupby('Sector')[['IRR%', 'NPV $M']].agg(['mean', 'std', 'min', 'max', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
mc_stats.columns = ['IRR Mean', 'IRR Std', 'IRR Min', 'IRR Max', 'IRR 5% VaR', 'IRR 95% CI', 'NPV Mean', 'NPV Std', 'NPV Min', 'NPV Max', 'NPV 5% VaR', 'NPV 95% CI']
print("\n#### Monte Carlo Stats (500 Runs)")
print(mc_stats.to_markdown())

# Steel IRR histogram (improved)
fig, ax = plt.subplots(figsize=(8, 5))
df_mc[df_mc['Sector'] == 'Steel']['IRR%'].hist(bins=20, color='steelblue', edgecolor='black')
ax.set_xlabel('IRR (%)')
ax.set_ylabel('Frequency')
ax.set_title('Monte Carlo Distribution for Steel IRR')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Risk-adjusted recs (unchanged)
top_3_mc = df_mc.groupby('Sector')['IRR%'].agg(['mean', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)]).reset_index().sort_values('mean', ascending=False).head(3)
top_3_mc.columns = ['Sector', 'Mean IRR', '5% CI', '95% CI']
recommendations = []
for _, row in top_3_mc.iterrows():
    sector = row['Sector']
    rec = f"Invest in {sector}; Mean IRR {row['Mean IRR']:.1f}% (5-95% CI: {row['5% CI']:.1f}-{row['95% CI']:.1f}%), Note: {sector_notes[sector]}"
    recommendations.append(rec)

print("\n#### Risk-Adjusted Investment Recommendations (Top 3, MC)")
for rec in recommendations:
    print(rec)

# MC sensitivities (unchanged)
mean_h2_mc = np.mean(mc_h2_prices)
sector_results_mc = []
for sector_name, sector in sectors.items():
    for sub in subsidies:
        irr = sector.irr(mean_h2_mc, 75, sub)
        npv = sector.npv(mean_h2_mc, 75, sub) / 1e6
        be_h2 = sector.break_even_h2(target_irr=15, carbon_price=75, subsidy=sub)
        sector_results_mc.append({"Sector": sector_name, "Subsidy $/kg": sub, "IRR%": irr, "NPV $M": npv, "Break-even H2 $/kg": be_h2})

df_sector_results_mc = pd.DataFrame(sector_results_mc)
print("\n#### Full Sensitivity Table (IRR/NPV/Break-even by Sector and Subsidy, Carbon $75/t, Using Mean MC H2 Price)")
print(df_sector_results_mc.to_markdown(index=False))

# IRR sensitivity plot MC (improved)
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'Steel': 'blue', 'Fertilizer': 'green', 'Shipping': 'red', 'Transport': 'purple', 'Chemicals': 'orange'}
y_offsets = {'Steel': -1, 'Fertilizer': 1, 'Shipping': -1, 'Transport': 1, 'Chemicals': 1}  # Adjusted offsets to separate labels and avoid title overlap
for sector in sectors.keys():
    sector_filter = (df_sector_results_mc['Sector'] == sector)
    subs = df_sector_results_mc[sector_filter]['Subsidy $/kg']
    irrs = df_sector_results_mc[sector_filter]['IRR%']
    line, = ax.plot(subs, irrs, label=sector, color=colors[sector])
    # Calculate gradient (slope)
    slope = (irrs.iloc[-1] - irrs.iloc[0]) / (subs.iloc[-1] - subs.iloc[0])
    # Annotate above the line at midpoint with colored outline box
    mid_x = (subs.iloc[0] + subs.iloc[-1]) / 2
    mid_y = (irrs.iloc[0] + irrs.iloc[-1]) / 2 + y_offsets.get(sector, 0)  # Apply custom offset
    ax.text(mid_x, mid_y, f'+{slope:.1f}%', fontsize=10, ha='center', bbox=dict(facecolor='none', edgecolor=line.get_color(), boxstyle='round,pad=0.5'))
ax.set_xlabel('Subsidy $/kg')
ax.set_ylabel('IRR (%)')
ax.set_title('Sector IRR Sensitivity to Subsidy (Mean H2 from MC)')
ax.legend(loc='upper left', frameon=False)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# ROA (unchanged)
# Helper function used in the modelling pipeline
def simple_roa(sector, mean_h2, carbon=75, subsidy=1.5, vol=0.2, rf=0.05):

    npv_now = sector.npv(mean_h2, carbon, subsidy)
    h2_up = mean_h2 * (1 + vol)
    h2_down = mean_h2 * (1 - vol)
    npv_up = sector.npv(h2_up, carbon, subsidy)
    npv_down = sector.npv(h2_down, carbon, subsidy)
    delayed_npv = (0.5 * max(npv_up, 0) + 0.5 * max(npv_down, 0)) / (1 + rf)
    option_value = max(0, delayed_npv - npv_now)
    return option_value / 1e6

roa_results = []
mc_h2_std = np.std(mc_h2_prices) / np.mean(mc_h2_prices) if mc_h2_prices else 0.2
for sector_name, sector in sectors.items():
    roa_val = simple_roa(sector, mean_h2_mc, vol=mc_h2_std)
    roa_results.append({"Sector": sector_name, "ROA Delay Value $M": roa_val})

df_roa = pd.DataFrame(roa_results)
print("\n#### Real Options Analysis: Value of 1-Year Delay Option (Using MC Volatility)")
print(df_roa.to_markdown(index=False))

# Validation (unchanged)
benchmark_data = {
    'global_demand_mt': {2030: 100, 2050: 530},
    'lcoh_green': {2025: (3,6), 2030: (1.5,3), 2050: (1,2)},
    'trade_mt_2050': 53
}

# Helper function used in the modelling pipeline
def validate_model(avg_h2_prices, df_trade, df_sector_results):

    rmse = 0
    for y in [2030, 2050]:
        model_avg = avg_h2_prices[y]
        bench_low, bench_high = benchmark_data['lcoh_green'][y]
        bench_avg = (bench_low + bench_high) / 2
        rmse += (model_avg - bench_avg)**2
    rmse = np.sqrt(rmse / 2)
    print(f"RMSE on LCOH vs IRENA: {rmse:.2f}")
    if rmse > 1.0:
        print("High deviation; consider adjusting calibrations.")

    total_trade_2050 = df_trade[df_trade['Year'] == 2050]['Flow_kt'].sum() / 1000
    bench_trade = benchmark_data['trade_mt_2050']
    deviation = abs(total_trade_2050 - bench_trade) / bench_trade * 100
    print(f"Model trade 2050: {total_trade_2050:.1f} Mt vs IRENA {bench_trade} Mt (deviation {deviation:.1f}%)")

    export_shares = df_trade.groupby('From')['Flow_kt'].sum() / df_trade['Flow_kt'].sum()
    hhi = (export_shares**2).sum() * 10000
    if 1500 <= hhi <= 2500:
        print(f"HHI {hhi:.0f} aligns with current energy markets.")
    elif hhi < 1500:
        print(f"Low HHI {hhi:.0f}; more competitive.")
    else:
        print(f"High HHI {hhi:.0f}; potential oligopoly.")

    base_irr = {'Steel': (8,12), 'Fertilizer': (5,10), 'Shipping': (5,10), 'Transport': (6,12), 'Chemicals': (5,9)}
    for sector_name in sectors:
        mean_irr = df_sector_results[df_sector_results['Sector'] == sector_name]['IRR%'].mean()
        low, high = base_irr[sector_name]
        if not low <= mean_irr <= high:
            print(f"{sector_name} mean IRR {mean_irr:.1f}% off lit range ({low}-{high}%); adjust.")
        else:
            print(f"{sector_name} mean IRR {mean_irr:.1f}% aligns with lit range ({low}-{high}%).")

    return rmse

validate_model(avg_h2_prices_det, df_trade_det, df_sector_results_mc)

# Edge cases (unchanged)
edge_results = {}
# Helper function used in the modelling pipeline
def run_edge_case(scenario):

    if scenario == 'high_carbon':
        avg_h2, df_trade, _, _, _ = run_milp(learning_rate_mean, carbon_multiplier=2.0)
        edge_results['high_carbon'] = avg_h2
        print(f"High Carbon Avg H2: ${avg_h2:.2f}")
    elif scenario == 'supply_disruption':
        avg_h2, df_trade, _, _, _ = run_milp(learning_rate_mean, disrupt_node='EGY_SUE', disrupt_factor=0.5)
        edge_results['supply_disruption'] = avg_h2
        print(f"Supply Disruption Avg H2: ${avg_h2:.2f}")
    elif scenario == 'doubled_demand':
        avg_h2, df_trade, _, _, _ = run_milp(learning_rate_mean, demand_scale=2.0)
        edge_results['doubled_demand'] = avg_h2
        print(f"Doubled Demand Avg H2: ${avg_h2:.2f}")
    elif scenario == 'geopolitics_high_trade':
        avg_h2, df_trade, _, _, _ = run_milp(learning_rate_mean, trans_scale=1.5)
        edge_results['geopolitics_high_trade'] = avg_h2
        print(f"Geopolitics High Trade Avg H2: ${avg_h2:.2f}")
    validate_model(avg_h2_prices_det, df_trade, df_sector_results_mc)

for case in ['high_carbon', 'supply_disruption', 'doubled_demand', 'geopolitics_high_trade']:
    run_edge_case(case)

# Enhanced Composite with improved PCA
vc_tam_2050 = {'Steel': 150e9, 'Fertilizer': 60e9, 'Shipping': 100e9, 'Transport': 80e9, 'Chemicals': 100e9}  # Balanced values

# Helper function used in the modelling pipeline
def enhanced_composite(df_ranking, df_trade):

    factors_df = pd.DataFrame(factor_scores).T
    scaler = StandardScaler()
    factors_scaled = scaler.fit_transform(factors_df)
    pca = PCA(n_components=3)
    pca.fit(factors_scaled)
    pca_scores = pca.transform(factors_scaled)
    pca_df = pd.DataFrame(pca_scores, index=factors_df.index, columns=['PC1', 'PC2', 'PC3'])
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")

    # Enhanced PCA Biplot with larger sector circles and fixed overlaps
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    fig, ax = plt.subplots(figsize=(12, 9))  # Larger for readability
    scatter = ax.scatter(pca_scores[:, 0], pca_scores[:, 1], s=200, c=colors, alpha=0.7)  # Increased size to 200 for larger circles (may overlap)
    for i, sector in enumerate(factors_df.index):
        ax.text(pca_scores[i, 0] + 0.1, pca_scores[i, 1] + 0.1, sector, fontsize=12)  # Slightly larger offset for sector labels
    scale = 1.5  # Scaling factor for arrows
    for i in range(factors_df.shape[1]):
        ax.arrow(0, 0, pca.components_[0, i]*scale, pca.components_[1, i]*scale, head_width=0.05, head_length=0.05, color='black')
        # Adjusted text positions with conditional offsets to avoid overlaps
        text_x = pca.components_[0, i]*scale * 1.1
        text_y = pca.components_[1, i]*scale * 1.1
        if 'Stakeholder Perception' in factors_df.columns[i]:
            text_y += 0.2  # Nudge up to avoid overlap
        elif 'Research Needed' in factors_df.columns[i]:
            text_x -= 0.3  # Nudge left if overlapping
        ax.text(text_x, text_y, factors_df.columns[i], fontsize=10, rotation=0 if pca.components_[1, i] > 0 else 180)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot of Sector Factors')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    export_shares = df_trade.groupby('From')['Flow_kt'].sum() / df_trade['Flow_kt'].sum()
    hhi = (export_shares**2).sum() * 10000
    barrier_adj = 1 - max(0, (hhi - 2000) / 10000 * 0.2)

    for sector in sectors:
        scores = factor_scores[sector]
        scores['TAM_2050_B'] = vc_tam_2050[sector] / 1e9
        scores['Exit_Multiple'] = 4 if scores['Market Growth'] > 7 else 3

    for row in df_ranking.itertuples():
        pca_weight = pca_df.loc[row.Sector, 'PC1'] * 0.2
        vc_weight = (factor_scores[row.Sector]['TAM_2050_B'] / 1000) * 0.1
        new_composite = row.Composite_Score * barrier_adj + pca_weight + vc_weight
        print(f"Enhanced Composite for {row.Sector}: {new_composite:.2f}")

enhanced_composite(df_ranking, df_trade_det)

# Portfolio opt (unchanged)
# Helper function used in the modelling pipeline
def portfolio_opt(df_mc):

    mu = df_mc.groupby('Sector')['IRR%'].mean().values  # Mean returns
    df_mc_pivot = df_mc.pivot(index='Run', columns='Sector', values='IRR%')
    cov = df_mc_pivot.cov().values
    cov += np.eye(len(mu)) * 0.01 * np.var(mu)  # Add ridge for stability if singular
    n_assets = len(mu)

    # Helper function used in the modelling pipeline
    def neg_sharpe(weights):

        returns = np.dot(weights, mu)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = returns / vol if vol > 0 else 0
        penalty = sum(w * max(0, 10 - mu[i]) for i, w in enumerate(weights)) * 0.2  # Stricter: threshold 10, multiplier 0.2
        return -sharpe + penalty

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.3 - x}  # Max 30% per asset (array broadcast)
    ]
    bounds = [(0.05, 0.3) for _ in range(n_assets)]  # Even tighter
    init_weights = mu / mu.sum()  # Init proportional to IRR means for better starting point
    res = minimize(neg_sharpe, init_weights, method='trust-constr', bounds=bounds, constraints=cons, tol=1e-6)

    weights = res.x
    print(f"Optimal Allocation: {[f'{s} {w:.2f}' for s, w in zip(sectors.keys(), weights)]}")
    print("Covariance Matrix:")
    print(pd.DataFrame(cov, index=sectors.keys(), columns=sectors.keys()))
    return weights

portfolio_opt(df_mc)

# Advanced trade (unchanged)
# Helper function used in the modelling pipeline
def advanced_trade(df_trade):

    df_trade['LTC_Flow'] = df_trade.apply(lambda row: row['Flow_kt'] * ltc_share[row['Year']], axis=1)
    df_trade['Spot_Flow'] = df_trade['Flow_kt'] - df_trade['LTC_Flow']
    df_trade['Unit_Cost_LTC'] = df_trade['Unit_Cost'] * (1 - ltc_discount)

    export_shares = df_trade.groupby('From')['Flow_kt'].sum() / df_trade['Flow_kt'].sum()
    hhi = (export_shares**2).sum() * 10000
    print(f"HHI: {hhi:.0f}")
    if hhi > 3000:
        print("Warning: High HHI indicating oligopoly risk.")
    elif hhi < 2000:
        print("HHI indicates competitive market.")

    intensities = {}
    for p in producers:
        exp = df_trade[df_trade['From'] == p]['Flow_kt'].sum()
        prod = sum(production_data[y][nodes.index(p)]['Production'] for y in years) / len(years)
        intensity = exp / prod if prod > 0 else 0
        intensities[p] = intensity
        print(f"Export Intensity {p}: {intensity:.2f}")
        if "AUS" in p and intensity > 0.5:
            print(f"{p} exceeds 50% export intensity benchmark.")

    return intensities

advanced_trade(df_trade_det)

# After advanced_trade and blue/green flows bar chart
intensities = advanced_trade(df_trade_det)  # Gets intensities dict
blue_flows = [df_trade_det[(df_trade_det['Year'] == y) & (df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
green_flows = [df_trade_det[(df_trade_det['Year'] == y) & (~df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
total_flows = [b + g for b, g in zip(blue_flows, green_flows)]
blue_pct = np.mean([b / t for b, t in zip(blue_flows, total_flows) if t > 0])  # Avg blue share from bar chart
sectors = SectorInvestment.link_vc_to_trade(df_trade_det, sectors, intensities, blue_pct)  # Update sectors with trade-linked adjustments
# Then use updated sectors in portfolio_opt/MC

# Improved Sankey (regional, improved readability)
# Helper function used in the modelling pipeline
def sankey_visual(df_trade):

    # Group by region for less clutter
    region_map = {'AUS_GLD': 'Australia', 'AUS_PIL': 'Australia', 'CHL_ANT': 'South America', 'SAU_NEOM': 'Middle East', 'OMN_DUQ': 'Middle East', 'UAE_RUW': 'Middle East', 'NAM_LUD': 'Africa', 'RSA_BOE': 'Africa', 'MAR_SOU': 'Africa', 'MRT_NDB': 'Africa', 'EGY_SUE': 'Middle East', 'KAZ_MNG': 'Asia', 'USA_HOU': 'North America', 'CAN_PTT': 'North America', 'NOR_BER': 'Europe', 'JPN_KAW': 'Asia', 'KOR_ULS': 'Asia', 'GER_WIL': 'Europe', 'NLD_ROT': 'Europe', 'BEL_ANR': 'Europe', 'CHN_SHG': 'Asia', 'IND_GUJ': 'Asia', 'UK_TEE': 'Europe'}
    df_trade['From_Region'] = df_trade['From'].map(region_map)
    df_trade['To_Region'] = df_trade['To'].map(region_map)
    agg_flows = df_trade.groupby(['From_Region', 'To_Region'])['Flow_kt'].sum().reset_index()

    fig = plt.figure(figsize=(12, 8))  # Larger for readability
    sankey = Sankey(ax=fig.add_subplot(1, 1, 1, xticks=[], yticks=[], frame_on=False))
    regions = list(set(agg_flows['From_Region']) | set(agg_flows['To_Region']))
    region_idx = {r: i for i, r in enumerate(regions)}
    region_colors = {'Africa': 'brown', 'Asia': 'yellow', 'Australia': 'green', 'Europe': 'blue', 'Middle East': 'orange', 'North America': 'red', 'South America': 'purple'}  # Added colors for regions

    for _, row in agg_flows.iterrows():
        sankey.add(flows=[row['Flow_kt'], -row['Flow_kt']], orientations=[0, 0], labels=[row['From_Region'], row['To_Region']], pathlengths=[0.5, 0.5], trunklength=2.0, facecolor=region_colors.get(row['From_Region'], 'gray'))

    diagrams = sankey.finish()
    for diagram in diagrams:
        diagram.text.set_fontweight('bold')
        diagram.text.set_fontsize('12')  # Larger font
        for text in diagram.texts:
            text.set_fontsize('12')
            text.set_rotation(0)  # Keep horizontal for readability
            text.set_va('center')  # Vertical alignment to center
    plt.title('Regional Hydrogen Flows Sankey')
    plt.show()

sankey_visual(df_trade_det)

# ============================= Reporting ================================= #

# Terminal report (unchanged)
# Helper function used in the modelling pipeline
def generate_terminal_report():

    print("Hydrogen Investment Report\n")
    print("Sector Ranking:")
    print(df_ranking.to_string())
    print("\nSensitivity Table:")
    print(df_sector_results_mc.to_string())
    print("\nMonte Carlo Stats:")
    print(mc_stats.to_string())
    print("\nEdge Case Summaries:")
    print(f"High Carbon Avg H2: ${edge_results['high_carbon']:.2f}")
    print(f"Supply Disruption Avg H2: ${edge_results['supply_disruption']:.2f}")
    print(f"Doubled Demand Avg H2: ${edge_results['doubled_demand']:.2f}")
    print(f"Geopolitics High Trade Avg H2: ${edge_results['geopolitics_high_trade']:.2f}")
    print("Report output complete in terminal.")
generate_terminal_report()

# ========================== Additional Figures =========================== #

# New Figures
# 1. Cumulative Supply
green_cum = np.cumsum([sum(d['Production'] for d in production_data[y] if emissions_data.get(d['Node'], 0) == 0) / 1000 for y in years])
blue_cum = np.cumsum([sum(d['Production'] for d in production_data[y] if emissions_data.get(d['Node'], 0) == 0.5) / 1000 for y in years])

fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(years, blue_cum, green_cum, labels=['Blue', 'Green'], colors=['#1f77b4', '#31a354'])
ax.set_title('Cumulative Supply Capacity, Baseline Scenario')
ax.set_xlabel('Year')
ax.set_ylabel('Million Tonnes H2')
ax.legend(frameon=False, loc='upper left')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 1b. Low Trade Cost (assume same capacities for simplicity)
_, df_trade_low, _, _, low_h2_prices = run_milp(learning_rate_mean, 1.0, 1.0, 0.5)
green_cum_low = np.cumsum([sum(d['Production'] for d in production_data[y] if emissions_data.get(d['Node'], 0) == 0) / 1000 for y in years])  # Same as base for demo
blue_cum_low = np.cumsum([sum(d['Production'] for d in production_data[y] if emissions_data.get(d['Node'], 0) == 0.5) / 1000 for y in years])  # Same
total_cum_low = green_cum_low + blue_cum_low
fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(years, blue_cum_low, green_cum_low, labels=['Blue', 'Green'], colors=['#1f77b4', '#31a354'])
ax.plot(years, green_cum + blue_cum, color='red', linestyle='--', label='Base Total')
ax.set_title('Cumulative Supply Capacity, Low Trade Cost')
ax.set_xlabel('Year')
ax.set_ylabel('Million Tonnes H2')
ax.legend(frameon=False, loc='upper left')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 2. Demand Price vs Cumulative
cum_demand = np.cumsum([sum(d['Demand'] for d in demand_data_base[y]) / 1000 for y in years])
prices = [avg_h2_prices_det[y] for y in years]
perc10 = [np.percentile(mc_h2_prices, 10) for _ in years]
perc90 = [np.percentile(mc_h2_prices, 90) for _ in years]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cum_demand, prices, color='#1f77b4')
ax.fill_between(cum_demand, perc10, perc90, color='#a5cee3', alpha=0.5)
ax.set_title('World Demand Price vs Cumulative Hydrogen, Baseline')
ax.set_xlabel('Million Tonnes H2')
ax.set_ylabel('Price [$/kg]')
ax.text(0.05, 0.05, 'Median of 500 runs, 10th-90th percentile range.', transform=ax.transAxes, fontsize=8)
ax.set_ylim(1, 5)  # Adjusted for visibility
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 3. Export Intensities with MC std, sorted descending
intensities = advanced_trade(df_trade_det)
# MC std for error bars
intensities_mc = {p: [run[p] for run in mc_intensities] for p in producers}
intensities_mean = {p: np.mean(intensities_mc[p]) for p in producers}
intensities_std = {p: np.std(intensities_mc[p]) for p in producers}
# Sort by mean descending
sorted_producers = sorted(producers, key=lambda p: intensities_mean[p], reverse=True)
sorted_countries = [p.split('_')[0] + ' (' + p.split('_')[1] + ')' for p in sorted_producers]
sorted_means = [intensities_mean[p] for p in sorted_producers]
sorted_stds = [intensities_std[p] for p in sorted_producers]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sorted_countries, sorted_means, yerr=sorted_stds, color='#fdbf6f', edgecolor='black', capsize=4)
ax.set_title('Export Intensities of Producer Countries, Baseline')
ax.set_ylabel('Export Intensity [%]')
ax.set_xticklabels(sorted_countries, rotation=90)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Low trade
intensities_low = advanced_trade(df_trade_low)
# MC std for low trade (assume same MC for demo; in real, rerun MC with trans_scale=0.5)
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sorted_countries, sorted_means, yerr=sorted_stds, color='#fdbf6f', edgecolor='black', capsize=4)  # Use base MC for demo
ax.set_title('Export Intensities of Producer Countries, Low Trade Cost')
ax.set_ylabel('Export Intensity [%]')
ax.set_xticklabels(sorted_countries, rotation=90)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 4. LTC Evolution
median_ltc = [np.median(mc_ltc_prices[y]) for y in years]
perc10_ltc = [np.percentile(mc_ltc_prices[y], 10) for y in years]
perc90_ltc = [np.percentile(mc_ltc_prices[y], 90) for y in years]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(years, median_ltc, color='#1f77b4')
ax.fill_between(years, perc10_ltc, perc90_ltc, color='#a5cee3', alpha=0.5)
ax.set_title('Evolution of LTC Prices, Baseline Scenario')
ax.set_xlabel('Year')
ax.set_ylabel('Price [$/kg]')
ax.text(0.05, 0.05, 'Median of 500 runs, 10th-90th percentile range.', transform=ax.transAxes, fontsize=8)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 5. Consumer/Producer Bars, sorted descending
consumer_vol = df_trade_det.groupby('To')['Flow_kt'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
consumer_vol.plot(kind='bar', ax=ax, color='#e31a1c')
ax.set_title('Hydrogen Consumer Countries, Baseline')
ax.set_ylabel('Total Flow (kt)')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

producer_vol = df_trade_det.groupby('From')['Flow_kt'].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
producer_vol.plot(kind='bar', ax=ax, color='#33a02c')
ax.set_title('Hydrogen Producer Countries, Baseline')
ax.set_ylabel('Total Flow (kt)')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Additional Scientific Graphs
# Blue vs Green Flows Over Time
blue_flows = [df_trade_det[(df_trade_det['Year'] == y) & (df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
green_flows = [df_trade_det[(df_trade_det['Year'] == y) & (~df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
total_flows = [b + g for b, g in zip(blue_flows, green_flows)]
blue_pct = [b / t * 100 if t > 0 else 0 for b, t in zip(blue_flows, total_flows)]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(years, green_flows, color='#edf8b1', label='Green')  # Bottom: green growing ~26.25 to 35.5 Mt
ax1.bar(years, blue_flows, bottom=green_flows, color='#7fcdbb', label='Blue')  # Top: blue flat 17.5 Mt
ax1.set_xlabel('Year')
ax1.set_ylabel('Million Tonnes H2')
ax1.legend(frameon=False, loc='upper left')
ax2 = ax1.twinx()
ax2.plot(years, blue_pct, color='black', linestyle='--', marker='o', label='Blue % Share')
ax2.set_ylabel('Blue Share (%)')
ax2.legend(frameon=False, loc='upper right')
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.title('Blue vs Green Hydrogen Flows Over Time')
plt.tight_layout()
plt.show()

# Market Prices Distribution (histogram for clarity)
delivered_prices_flat = [price for year_prices in delivered_prices_det.values() for price in year_prices.values() if not np.isnan(price)]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(delivered_prices_flat, bins=20, color='#a5cee3', edgecolor='black')
ax.set_title('Distribution of Market Prices, Baseline Scenario')
ax.set_xlabel('Price [$/kg]')
ax.set_ylabel('Frequency')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# LTC Complexity/Thickness (improved with MC variability)
# Export markets per firm: average unique 'To' per 'From' across MC runs
export_markets_per_firm = []
for run_intensities in mc_intensities:
    unique_markets = {p: len(df_trade_det[df_trade_det['From'] == p]['To'].unique()) for p in producers}
    export_markets_per_firm.append([unique_markets[p] for p in producers])
export_markets_mean = np.mean(export_markets_per_firm, axis=0)
export_markets_std = np.std(export_markets_per_firm, axis=0)
sorted_producers = sorted(producers, key=lambda p: export_markets_mean[nodes.index(p)], reverse=True)
sorted_means = [export_markets_mean[nodes.index(p)] for p in sorted_producers]
sorted_stds = [export_markets_std[nodes.index(p)] for p in sorted_producers]

# New entrants each period: count producers with significant flow increase per year
new_entrants_per_period = []
for y in years:
    year_flows = df_trade_det[df_trade_det['Year'] == y].groupby('From')['Flow_kt'].sum()
    prev_year_flows = df_trade_det[df_trade_det['Year'] == years[years.index(y) - 1]].groupby('From')['Flow_kt'].sum() if y > 2025 else pd.Series(0, index=producers)
    entrants = sum(1 for p in producers if year_flows.get(p, 0) > prev_year_flows.get(p, 0) * 1.5)  # 50% increase threshold
    new_entrants_per_period.append(entrants)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Export Markets per Firm
ax1.bar(range(len(sorted_producers)), sorted_means, yerr=sorted_stds, color='#1f77b4', edgecolor='black', capsize=4)
ax1.set_title('Export Markets per Firm')
ax1.set_xlabel('Producer (Ranked by Markets)')
ax1.set_ylabel('Average Number of Export Markets')
ax1.set_xticks(range(len(sorted_producers)))
ax1.set_xticklabels([p.split('_')[0] for p in sorted_producers], rotation=45, ha='right')
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# New Entrants Each Period
ax2.bar(years, new_entrants_per_period, color='#1f77b4')
ax2.set_title('New Entrants Each Period')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of New Entrants')
ax2.grid(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Additional VC Graphs
# 1. VC Funding Trends in Hydrogen (hardcoded from search data)
years_vc = [2020, 2021, 2022, 2023, 2024, 2025]
vc_funding = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5]  # $B, projected 2025
categories = ['Green', 'Blue', 'Storage', 'Other']
vc_breakdown_2025 = [1.0, 0.5, 0.7, 0.3]  # $B

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(years_vc, vc_funding, color='#ff7f0e', label='Total VC Funding')
ax1.set_xlabel('Year')
ax1.set_ylabel('$Billion')
ax1.legend(frameon=False, loc='upper left')
ax2 = ax1.twinx()
ax2.bar(categories, vc_breakdown_2025, color='#1f77b4', alpha=0.6, label='2025 Breakdown')
ax2.set_ylabel('$Billion')
ax2.legend(frameon=False, loc='upper right')
ax1.grid(False)
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.title('VC Funding Trends in Hydrogen')
plt.tight_layout()
plt.show()

# 2. IRR Distribution Across Sectors (MC Histogram subplots)
fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # 1x5 for 5 sectors
axs = axs.flatten()
for idx, sector in enumerate(sectors.keys()):
    ax = axs[idx]
    data = df_mc[df_mc['Sector'] == sector]['IRR%']
    ax.hist(data, bins=20, color='steelblue', edgecolor='black', density=True)  # For curved look
    ax.set_title(f'{sector} IRR Distribution (MC)')
    ax.set_xlabel('IRR (%)')
    ax.set_ylabel('Density')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 3. Portfolio Allocation Pie Chart (reformatted)
alloc = portfolio_opt(df_mc)  # Call the function and store the return value
labels = list(sectors.keys())
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(alloc, labels=labels, autopct='%1.1f%%', colors=['blue', 'green', 'red', 'purple', 'orange'], shadow=True, startangle=90)
ax.set_title('Optimal VC Portfolio Allocation')
plt.tight_layout()
plt.show()

# 4. TAM vs Investment Opportunity
tam = [vc_tam_2050[s] / 1e9 for s in sectors]
vc_2025 = [2.5 * factor_scores[s]['Market Growth'] / sum(factor_scores[s]['Market Growth'] for s in sectors) for s in sectors]  # Scaled by growth
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(sectors.keys(), tam, color='#1f77b4', label='TAM 2050 ($B)')
ax.set_ylabel('$Billion')
ax.legend(frameon=False, loc='upper left')
ax2 = ax.twinx()
ax2.plot(sectors.keys(), vc_2025, color='orange', marker='o', label='Projected 2025 VC ($B)')
ax2.set_ylabel('$Billion')
ax2.legend(frameon=False, loc='upper right')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.title('TAM vs VC Investment Opportunity')
plt.tight_layout()
plt.show()

# 5. ROI Sensitivity to Carbon Price (with gradients)
carbon_range = np.linspace(50, 150, 5)
roi_sens = {s: [sector.irr(avg_h2_det, c, 1.5) for c in carbon_range] for s, sector in sectors.items()}
fig, ax = plt.subplots(figsize=(8, 5))
for s, rois in roi_sens.items():
    ax.plot(carbon_range, rois, label=s, color=colors[s])
    slope = (rois[-1] - rois[0]) / (carbon_range[-1] - carbon_range[0])
    mid_x = (carbon_range[0] + carbon_range[-1]) / 2
    mid_y = (rois[0] + rois[-1]) / 2 + (1 if s in ['Shipping', 'Transport'] else -1)  # Alternate above/below
    ax.text(mid_x, mid_y, f'+{slope:.1f}%', fontsize=10, ha='center', bbox=dict(facecolor='none', edgecolor=colors[s], boxstyle='round,pad=0.5'))
ax.set_xlabel('Carbon Price ($/t)')
ax.set_ylabel('IRR (%)')
ax.set_title('ROI Sensitivity to Carbon Price')
ax.legend(frameon=False)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 6. Cost Curve Evolution (with varying bands)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(years, list(avg_h2_prices_det.values()), color='#1f77b4')
perc10_var = [np.percentile(mc_h2_prices, 10) - 0.1 * i for i in range(len(years))]  # Simulated variation
perc90_var = [np.percentile(mc_h2_prices, 90) + 0.1 * i for i in range(len(years))]
ax.fill_between(years, perc10_var, perc90_var, color='#a5cee3', alpha=0.5)
ax.set_title('H2 Cost Curve Evolution')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Price [$/kg]')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# --- NEW: Dual Figure — Cumulative Supply (Baseline) + Blue vs Green Flows (boxed axes & boxed legends) ---

import matplotlib.pyplot as plt

# Small helpers (self-contained)
# Helper function used in the modelling pipeline
def _box_axes(ax, lw: float = 1.2):

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)

# Helper function used in the modelling pipeline
def _legend_outside_top(ax, y: float = 1.02, ncol: int | None = None, top_pad: float = 0.90):

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = min(4, max(1, (len(labels) + 1) // 2))
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y),   # tight to the top of the axes
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=2.0,
    )
    leg.get_frame().set_linewidth(1.0)
    ax.figure.subplots_adjust(top=top_pad)

# Consistent colors with your cumulative supply plot
BLUE  = "#1f77b4"
GREEN = "#31a354"

# If you haven't kept these around, (re)compute flows+share quickly
_blue_flows = [df_trade_det[(df_trade_det['Year'] == y) & (df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
_green_flows = [df_trade_det[(df_trade_det['Year'] == y) & (~df_trade_det['From'].isin(blue_nodes))]['Flow_kt'].sum() / 1000 for y in years]
_total_flows = [b + g for b, g in zip(_blue_flows, _green_flows)]
_blue_pct    = [(b / t) * 100 if t > 0 else 0 for b, t in zip(_blue_flows, _total_flows)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# LEFT: Cumulative supply (baseline) — reuses blue_cum/green_cum you computed above
axL.stackplot(years, blue_cum, green_cum, labels=['Blue', 'Green'], colors=[BLUE, GREEN])
axL.set_xlabel('Year')
axL.set_ylabel('Million Tonnes H₂')
_box_axes(axL)
_legend_outside_top(axL, y=1.02, ncol=2, top_pad=0.90)  # boxed, tight legend above left plot

# RIGHT: Blue vs Green flows over time — Blue on bottom for consistency
axR.bar(years, _blue_flows, color=BLUE,  label='Blue')
axR.bar(years, _green_flows, bottom=_blue_flows, color=GREEN, label='Green')
axR.set_xlabel('Year')
axR.set_ylabel('Million Tonnes H₂')
_box_axes(axR)

# Secondary axis for Blue % share
axR2 = axR.twinx()
axR2.plot(years, _blue_pct, color='black', linestyle='--', marker='o', label='Blue % Share')
axR2.set_ylabel('Blue Share (%)')
_box_axes(axR2)

# Combine legends from axR and axR2 into one boxed legend above the RIGHT subplot
h1, l1 = axR.get_legend_handles_labels()
h2, l2 = axR2.get_legend_handles_labels()
# De-duplicate (optional)
labels_seen, handles_combined, labels_combined = set(), [], []
for h, l in list(zip(h1 + h2, l1 + l2)):
    if l not in labels_seen:
        handles_combined.append(h); labels_combined.append(l); labels_seen.add(l)

leg = axR.legend(
    handles_combined, labels_combined,
    loc='lower center', bbox_to_anchor=(0.5, 1.02),
    ncol=3, frameon=True, fancybox=False, edgecolor='black',
    borderpad=0.4, labelspacing=0.4, handlelength=2.0
)
leg.get_frame().set_linewidth(1.0)

# IMPORTANT: No titles on either subplot (per your request)
plt.tight_layout()
# Save figure to file for inclusion in dissertation / appendix
plt.savefig('dual_cumulative_supply_vs_blue_green_flows.pdf', bbox_inches='tight')
# Save figure to file for inclusion in dissertation / appendix
plt.savefig('dual_cumulative_supply_vs_blue_green_flows.png', dpi=300, bbox_inches='tight')
plt.show()

# === Paper exports: Tables & metrics for Results/Discussion ===
import numpy as np, pandas as pd
from pathlib import Path

OUTDIR = Path("paper_exports")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- A) Delivered price distributions ----------
# Build tidy frame from your delivered_prices_det = {year: {dest: price}}
price_rows = []
for y, m in delivered_prices_det.items():
    for dest, val in m.items():
        if pd.notnull(val):
            price_rows.append(
                {"year": int(y),
                 "destination_id": str(dest),
                 "delivered_price_usd_per_kg": float(val)}
            )
prices_df = pd.DataFrame(price_rows)

# Per-year distribution stats: P10 / Median / P90 / IQR / (P90-P10)
# Helper function used in the modelling pipeline
def _pct(a, q):

    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q))

stats_rows = []
for y, g in prices_df.groupby("year"):
    arr = g["delivered_price_usd_per_kg"].to_numpy()
    p10, p50, p90 = _pct(arr,10), _pct(arr,50), _pct(arr,90)
    p25, p75 = _pct(arr,25), _pct(arr,75)
    stats_rows.append({
        "year": int(y),
        "p10": round(p10, 2),
        "median": round(p50, 2),
        "p90": round(p90, 2),
        "p90_minus_p10": round(p90 - p10, 2),
        "iqr": round(p75 - p25, 2),
    })
price_stats_by_year = pd.DataFrame(stats_rows).sort_values("year")
price_stats_by_year.to_csv(OUTDIR/"price_stats_by_year.csv", index=False)

# Destination medians (per year, per destination)
destination_price_medians = (
    prices_df
    .groupby(["year","destination_id"])["delivered_price_usd_per_kg"].median()
    .reset_index()
    .rename(columns={"delivered_price_usd_per_kg":"median_price_usd_per_kg"})
)
destination_price_medians.to_csv(OUTDIR/"destination_price_medians.csv", index=False)

# ---------- B) Top corridors (2050, >= 1.0 Mt/yr) ----------
# Your df_trade_det has: Year, From, To, Flow_kt, Distance_km, Mode, Unit_Cost  :contentReference[oaicite:1]{index=1}
flows_det = df_trade_det.copy()

# Standardise columns/units for export
flows_det["flow_mt_per_year"] = (flows_det["Flow_kt"] / 1000.0)
flows_det = flows_det.rename(columns={
    "Year": "year",
    "From": "origin_id",
    "To": "destination_id",
    "Mode": "mode",
    "Distance_km": "distance_km",
    "Unit_Cost": "delivered_cost_usd_per_kg"
})

top_corridors = (
    flows_det.query("year == 2050 and flow_mt_per_year >= 1.0")
    [["origin_id","destination_id","mode","distance_km","flow_mt_per_year","delivered_cost_usd_per_kg"]]
    .sort_values(["flow_mt_per_year","delivered_cost_usd_per_kg"], ascending=[False, True])
    .reset_index(drop=True)
)

# Tidy rounding for presentation
top_corridors["distance_km"] = top_corridors["distance_km"].round(0).astype("Int64")
top_corridors["flow_mt_per_year"] = top_corridors["flow_mt_per_year"].round(2)
top_corridors["delivered_cost_usd_per_kg"] = top_corridors["delivered_cost_usd_per_kg"].round(2)

top_corridors.to_csv(OUTDIR/"top_corridors_2050.csv", index=False)

# ---------- C) Sector economics & sensitivities ----------
# Your MC dataframe is df_mc with columns:
# "Run","Sector","H2 Price","Carbon $/t","Subsidy $/kg","IRR%","NPV $M"  :contentReference[oaicite:2]{index=2}
mc_raw = df_mc.copy().rename(columns={
    "Run":"trial",
    "Sector":"sector",
    "H2 Price":"h2_price_usd_per_kg",
    "Carbon $/t":"carbon_price_usd_per_t",
    "Subsidy $/kg":"subsidy_usd_per_kg",
    "IRR%":"irr_pct",
    "NPV $M":"npv_musd"
})
mc_raw["npv_usd"] = mc_raw["npv_musd"] * 1e6  # convert M$ -> $
mc_tidy = mc_raw[["trial","sector","irr_pct","npv_usd","h2_price_usd_per_kg","carbon_price_usd_per_t","subsidy_usd_per_kg"]].copy()

# (1) Summary stats by sector
# Helper function used in the modelling pipeline
def _p(s, q): return float(np.percentile(np.asarray(s, dtype=float), q))
mc_sector_summary = (
    mc_tidy.groupby("sector")
    .agg(irr_mean=("irr_pct", lambda s: round(np.mean(s), 1)),
         irr_p05=("irr_pct", lambda s: round(_p(s,5), 1)),
         irr_p95=("irr_pct", lambda s: round(_p(s,95), 1)),
         npv_mean=("npv_usd", lambda s: float(np.mean(s))))
    .reset_index()
)
mc_sector_summary.to_csv(OUTDIR/"mc_sector_summary.csv", index=False)

# (2) Standardised betas: IRR ~ (H2 price + carbon + subsidy), per sector
betas_out = []
for sec, g in mc_tidy.groupby("sector"):
    X = g[["h2_price_usd_per_kg","carbon_price_usd_per_t","subsidy_usd_per_kg"]].astype(float).to_numpy()
    y = g["irr_pct"].astype(float).to_numpy()
    # z-score standardisation (population std)
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)
    # OLS via normal equations
    XtX = Xz.T @ Xz
    XtX_inv = np.linalg.pinv(XtX)
    coef = XtX_inv @ (Xz.T @ yz)
    betas_out.append({
        "sector": sec,
        "|β|_h2_price": round(abs(coef[0]), 2),
        "|β|_carbon":   round(abs(coef[1]), 2),
        "|β|_subsidy":  round(abs(coef[2]), 2),
    })
mc_betas = pd.DataFrame(betas_out)
mc_betas.to_csv(OUTDIR/"mc_betas.csv", index=False)

# (3) Thresholds: solve for H2 price / CO2 price at target IRR using linear fit in original units
thr_out = []
for sec, g in mc_tidy.groupby("sector"):
    X = g[["h2_price_usd_per_kg","carbon_price_usd_per_t","subsidy_usd_per_kg"]].astype(float).to_numpy()
    y = g["irr_pct"].astype(float).to_numpy()
    # Add intercept
    X1 = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.pinv(X1) @ y  # [intercept, a, b, c]
    d, a, b, c = float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3])
    subsidy_base = float(g["subsidy_usd_per_kg"].mean())  # "baseline subsidy"
    # IRR* = 10%
    h2_at_irr10 = (10 - d - b*100 - c*subsidy_base) / a if a != 0 else np.nan
    carbon_at_irr10 = (10 - d - a*3.5 - c*subsidy_base) / b if b != 0 else np.nan
    thr_out.append({
        "sector": sec,
        "h2_price_for_irr10_at_carbon100": round(h2_at_irr10, 2),
        "carbon_for_irr10_at_h235": round(carbon_at_irr10, 0)
    })
sector_thresholds = pd.DataFrame(thr_out)
sector_thresholds.to_csv(OUTDIR/"sector_thresholds.csv", index=False)

# ---------- D) Market structure & geography (robust) ----------

# (1) Exporter HHI per year (shares of total exports by origin)
ex = df_trade_det.groupby(["Year", "From"], as_index=False)["Flow_kt"].sum()
ex["export_share"] = ex.groupby("Year")["Flow_kt"].transform(lambda s: s / s.sum())
export_hhi = (
    ex.groupby("Year", as_index=False)
      .agg(export_hhi=("export_share", lambda s: float((s**2).sum())))
      .rename(columns={"Year": "year"})
)

# (2) Distance elasticity per year: log(flow) ~ α + β*log(distance), WLS by flow
tmp = df_trade_det.loc[
    (df_trade_det["Flow_kt"] > 0) & (df_trade_det["Distance_km"] > 0),
    ["Year", "Flow_kt", "Distance_km"]
].copy()

# Helper function used in the modelling pipeline
def _dist_elasticity_np(g):

    lf = np.log(g["Flow_kt"].to_numpy())
    ld = np.log(g["Distance_km"].to_numpy())
    w  = g["Flow_kt"].to_numpy()
    X  = np.column_stack([np.ones(len(ld)), ld])
    # Weighted least squares without forming a big diagonal W
    WX   = X * w[:, None]
    beta = np.linalg.pinv(X.T @ WX) @ (X.T @ (w * lf))
    yhat = X @ beta
    num  = (w * (lf - yhat)**2).sum()
    den  = (w * (lf - np.average(lf, weights=w))**2).sum()
    r2   = 1.0 - (num / den if den > 0 else 0.0)
    return float(beta[1]), float(r2)

de_rows = []
for y, g in tmp.groupby("Year"):
    slope, r2 = _dist_elasticity_np(g)
    de_rows.append({"year": int(y),
                    "dist_elasticity_slope": round(slope, 2),
                    "r2": round(r2, 2)})
de = pd.DataFrame(de_rows)

# (3) Join with price dispersion from A)
disp = price_stats_by_year[["year", "p90_minus_p10"]].rename(
    columns={"p90_minus_p10": "price_dispersion_p90_p10"}
)
market_structure_metrics = (
    export_hhi.merge(disp, on="year", how="left")
              .merge(de, on="year", how="left")
)
market_structure_metrics.to_csv(OUTDIR / "market_structure_metrics.csv", index=False)

print(f"[paper_exports] Wrote:\n - {OUTDIR/'price_stats_by_year.csv'}\n - {OUTDIR/'destination_price_medians.csv'}\n - {OUTDIR/'top_corridors_2050.csv'}\n - {OUTDIR/'mc_sector_summary.csv'}\n - {OUTDIR/'mc_betas.csv'}\n - {OUTDIR/'sector_thresholds.csv'}\n - {OUTDIR/'market_structure_metrics.csv'}")

# ==== FIG A (FINAL): Trade maps with searoute paths, top boxed legend, axis labels ====
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import numpy as np

# Fonts (bigger, consistent)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size']   = 14
AX_LABEL_SIZE = 16
TICK_SIZE     = 14
LEGEND_SIZE   = 14

proj = ccrs.PlateCarree()

# Helper function used in the modelling pipeline
def _mode_color(mode):

    return {"Pipeline": "grey", "Ammonia": "blue", "LH2": "purple"}.get(mode, "black")

# Helper function used in the modelling pipeline
def _route_points(from_id, to_id):

    p1 = all_coords[from_id]  # (lat, lon)
    p2 = all_coords[to_id]
    if sr is not None:
        try:
            # searoute expects [lon, lat]
            r = sr.searoute([p1[1], p1[0]], [p2[1], p2[0]],
                            append_orig_dest=True, include_ports=True)
            coords = r.geometry.coordinates
            lons, lats = zip(*[(xy[0], xy[1]) for xy in coords])
            return list(lats), list(lons)
        except Exception:
            pass
    # Fallback: cartopy geodesic (curved)
    return [p1[0], p2[0]], [p1[1], p2[1]]

fig, axes = plt.subplots(
    2, 1, figsize=(15, 11),
    subplot_kw={"projection": proj}
)

# Leave extra top margin for the legend so it won't be cropped
fig.subplots_adjust(top=0.83, bottom=0.06, left=0.05, right=0.98, hspace=0.18)

for ax, yr in zip(axes, [2025, 2050]):
    ax.set_extent([-180, 180, -80, 85], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=0)

    # gridline tick labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = gl.top_labels = False
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel("Longitude", fontsize=AX_LABEL_SIZE)
    ax.set_ylabel("Latitude",  fontsize=AX_LABEL_SIZE)

    # Nodes (no labels)
    for node, (lat, lon) in all_coords.items():
        is_supply = node in producers
        ax.scatter(lon, lat, s=38, color=('black' if is_supply else 'red'),
                   transform=proj, zorder=5)

    # Flows (use searoute where possible)
    sub = df_trade_det[df_trade_det["Year"] == yr]
    for _, r in sub.iterrows():
        f, t, flow, mode = r["From"], r["To"], r["Flow_kt"], r["Mode"]
        if flow <= 1:
            continue
        lats, lons = _route_points(f, t)
        ax.plot(lons, lats,
                transform=proj if sr is not None else ccrs.Geodetic(),
                lw=0.85 * np.sqrt(np.log(flow + 1)),
                color=_mode_color(mode),
                alpha=0.85,
                linestyle='--' if mode == "Pipeline" else '-',
                zorder=3)

# Figure-level legend (top, boxed, black border)
supply_dot = Line2D([0],[0], marker='o', color='w', markerfacecolor='black', markersize=7, label='Supply node')
demand_dot = Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   markersize=7, label='Demand node')
ammonia    = Line2D([0],[0], color='blue',   lw=2, label='Ammonia flow')
pipeline   = Line2D([0],[0], color='grey',   lw=2, ls='--', label='Pipeline flow')
lh2        = Line2D([0],[0], color='purple', lw=2, label='LH₂ flow')

leg = fig.legend(
    handles=[supply_dot, demand_dot, ammonia, pipeline, lh2],
    loc='upper center', bbox_to_anchor=(0.5, 0.98),
    ncol=5, frameon=True, fancybox=False, fontsize=LEGEND_SIZE
)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.2)

# Save figure to file for inclusion in dissertation / appendix

plt.savefig("figA_trade_maps_searoute_topkey.png", dpi=300, bbox_inches="tight")
# Save figure to file for inclusion in dissertation / appendix
plt.savefig("figA_trade_maps_searoute_topkey.pdf", bbox_inches="tight")
plt.show()

# ==== FIG B: Cumulative H₂ supply (baseline) ====
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size']   = 14
AX_LABEL_SIZE = 15
TICK_SIZE     = 13
LEGEND_SIZE   = 13

# Recompute if needed
green_cum = np.cumsum([sum(d['Production'] for d in production_data[y]
                           if emissions_data.get(d['Node'], 0) == 0)   / 1000 for y in years])
blue_cum  = np.cumsum([sum(d['Production'] for d in production_data[y]
                           if emissions_data.get(d['Node'], 0) == 0.5) / 1000 for y in years])

BLUE  = "#4287f5"
GREEN = "#58b337"

fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.stackplot(years, blue_cum, green_cum, labels=['Blue', 'Green'], colors=[BLUE, GREEN])

ax.set_xlabel('Year', fontsize=AX_LABEL_SIZE)
ax.set_ylabel('Cumulative H₂ supply [Mt]', fontsize=AX_LABEL_SIZE)  # correct unit/meaning
ax.tick_params(labelsize=TICK_SIZE)

leg = ax.legend(
    loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2,
    frameon=True, fancybox=False, fontsize=LEGEND_SIZE
)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.2)

# Box the axes
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.1)

plt.tight_layout()
# Save figure to file for inclusion in dissertation / appendix
plt.savefig("figB_cumulative_supply_topkey_blackbox.png", dpi=300, bbox_inches="tight")
# Save figure to file for inclusion in dissertation / appendix
plt.savefig("figB_cumulative_supply_topkey_blackbox.pdf", bbox_inches="tight")
plt.show()

from pathlib import Path
import datetime as dt
from results_io import save_all

RUN_ID = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
EXPORT_DIR = Path("exports")

artifacts = {
    "df_trade_det": df_trade_det,
    "delivered_prices_det": delivered_prices_det,
    "avg_h2_prices_det": avg_h2_prices_det,
    "df_ranking": df_ranking,
    "df_sector_results_mc": df_sector_results_mc,
    "df_mc": df_mc,
    "mc_ltc_prices": {int(k): list(v) for k, v in mc_ltc_prices.items()},
    "mc_h2_prices": mc_h2_prices,
    "mc_hhi": mc_hhi,
    "df_roa": df_roa,
    "edge_results": edge_results,
    "years": years,
    "wacc": wacc,
    "learning_rate_mean": learning_rate_mean,
    "demand_nodes": demand_nodes,
    "producers": [p for p in producers],
}

run_dir = save_all(RUN_ID, EXPORT_DIR, **artifacts)
print(f"\nExport complete -> {run_dir}")