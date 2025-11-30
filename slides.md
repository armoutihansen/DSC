# CitiBike Data Challenge  

**Jesper Armouti-Hansen**  
Data Science Case Study

---

## Agenda

1. CitiBike Overview  
2. Data Analysis (usage patterns)  
3. Risk Analysis (collisions × trips)  
4. Net Flow Prediction (concept & next steps)  
5. Conclusion & Business Opportunities  

*(Today’s focus: 2 & 3. Sections 1, 4, and 5 are still evolving.)*

---

# 1. CitiBike Overview

---

## CitiBike in a Nutshell

- **Largest bike sharing system in the US**  
- ~**12,000+ bikes**, **hundreds of stations** across NYC  
- Real alternative to:
  - Subway for short trips
  - Taxis / ride-hailing for congested corridors  
- Rich data:
  - Start/end station, time stamps
  - Bike type (classic / electric)
  - Rider type (member / casual)

---

## Data Sources Used

- **CitiBike trip data** (Jan 2023 – Oct 2025)
  - Granular trip-level information
  - Used for **demand patterns** and **exposure** in risk analysis  
- **NYPD collision data**
  - Contains accidents involving cyclists
  - Used to build a **station-level risk measure**  
- Geo-information for stations
  - Latitude / longitude
  - Used to assign collisions to stations

---

# 2. Data Analysis

---

## Daily Usage Over Time

- Objective: understand overall **growth, seasonality, and volatility** in demand.
- Data: daily trip counts across all stations.

<img src="presentation/assets/img/daily_citibike_usage.png" width="70%">

---

## Key Patterns in Daily Usage

- **Strong seasonal pattern**:
  - Higher usage in warmer months, lower in winter.
- **Week-to-week variation**:
  - Weather and holidays likely play a role.
- This motivates:
  - Incorporating **calendar** and **(lagged) weather** features later.
  - Designing operations & insurance products that are **seasonally aware**.

---

## Bike Type Mix

- CitiBike fleet includes **classic bikes** and **e-bikes**.
- Plot: share of rides by bike type.

<img src="presentation/assets/img/bike_type_share.png" width="65%">

**Takeaways:**
- E-bikes account for a **substantial and growing share** of trips.
- Important for:
  - **Risk:** higher speeds may change accident severity.
  - **Pricing:** e-bikes might justify differentiated insurance premiums.

---

## Membership vs. Casual Riders

- CitiBike distinguishes **members** and **casual (one-off) users**.

<img src="presentation/assets/img/membership_type_share.png" width="65%">

**Implications:**
- Members generate a large share of trips → **high-value segment**.
- Casual riders may have:
  - Different **risk profiles** (less familiar with system/traffic).
  - Different sensitivity to **per-ride insurance offers**.

---

## Weekday vs Weekend Patterns

- Compare usage distribution between **weekdays** and **weekends**.

<img src="presentation/assets/img/weekday_vs_weekend_share.png" width="70%">

**Observations:**
- Weekdays: commuting & routine trips dominate.
- Weekends: more **leisure** and **tourist** usage.
- Suggests:
  - Different **temporal risk profiles** (e.g., commuting peaks vs. tourist hotspots).
  - Potential for **time-/day-specific pricing or messaging**.

---

## Hourly Usage Profile

- How trips distribute across the **24 hours** of the day.

<img src="presentation/assets/img/hourly_usage_share.png" width="70%">

**Insights:**
- Clear **peak hours** in the morning and evening.
- Off-peak hours show a long tail of lower but non-zero demand.
- For risk and operations:
  - Match **temporal risk** against this **temporal exposure**.
  - Identify hours where risk per trip is unusually high/low.

---

## Trip Duration and Distance

- Joint distribution of **trip duration** and **approximate distance**.

<img src="presentation/assets/img/duration_distance_distribution.png" width="70%">

**Interpretation:**
- Majority of trips are **short in both time and distance**.
- Long-duration / long-distance tail:
  - Potentially different rider types (leisure/tourists).
  - May face different **exposure to traffic risk**.
- For insurance:
  - Allows thinking about **risk per minute** or **risk per km**, not just per trip.

---

## Net Flow Across Stations

- For each station and time period, define **net flow** as:
  \[
  \text{net\_flow} = \text{arrivals} - \text{departures}
  \]

<img src="presentation/assets/img/net_flow_analysis.png" width="70%">

**Insights:**
- Most stations fluctuate around **balanced net flow**.
- Some stations show **persistent surpluses or deficits**:
  - Indicate **rebalancing needs**.
  - Motivate the **Net Flow Prediction** section later.

---

# 3. Risk Analysis

---

## Goal of the Risk Analysis

- Combine **collision data** and **CitiBike trips** to:
  1. Quantify **station-level risk per trip**.
  2. Understand how risk varies across:
     - **Stations**
     - **Time of day**
     - **Day of week**
  3. Identify:
     - **High-risk stations / conditions**.
     - Use cases for **micro-insurance** and **user warnings**.

---

## Assigning Collisions to Stations

To connect collisions to CitiBike usage:

1. Use station coordinates and crash locations (lat/long).
2. Build a **BallTree** with **Haversine distance** on station locations.
3. For each crash:
   - Find the **nearest station** within **300 meters**.
   - If no station is within 300m, the crash is **not assigned**.
4. This yields a set of **collisions per station**.

**Why this is reasonable:**
- 300m is a **short walking distance** in NYC.
- Crashes near a station proxy **local cyclist risk**, even if not every crash involves a CitiBike.

---

## Defining Severity of a Collision

Each crash is summarized by a **severity score**:

\[
\text{severity}_i = w_1 \cdot \text{cyclists\_injured}_i
+ w_2 \cdot \text{cyclists\_killed}_i
+ w_3 \cdot \text{others\_injured}_i
+ w_4 \cdot \text{others\_killed}_i
\]

- Weights \( w_1, \dots, w_4 \) encode the **relative importance** of injuries vs fatalities, cyclists vs others.
- This reduces each crash \( i \) to a **single numeric severity**.

---

## Exposure and Raw Risk Measure

For a given station \( s \) and time aggregation \( t \):

- **Hazard (numerator)**:
  \[
  H_{s,t} = \sum_{i \in \mathcal{C}_{s,t}} \text{severity}_i
  \]

- **Exposure (denominator)**:
  \[
  E_{s,t} = \text{number of CitiBike trips at station } s \text{ in period } t
  \]

- **Raw risk per trip**:
  \[
  R^{\text{raw}}_{s,t} = \frac{H_{s,t}}{E_{s,t} + \varepsilon}
  \]

with \( \varepsilon > 0 \) a small constant to avoid division by zero.

**Interpretation:**  
\( R^{\text{raw}}_{s,t} \) = **severity per CitiBike trip** for station \( s \), time \( t \).

---

## Empirical Bayes Smoothing

Raw risk is noisy, especially for **low-exposure** station×time cells.

We use **Empirical Bayes (EB)** smoothing:

\[
R^{\text{EB}}_{s,t}
= \lambda_{s,t} \, R^{\text{raw}}_{s,t}
+ (1 - \lambda_{s,t}) \, \mu
\]

- \( \mu \): overall **average severity per trip** across all stations and times.
- \( \lambda_{s,t} \in [0,1] \): **shrinkage factor**:
  - Close to 1 when **exposure is high**.
  - Close to 0 when **exposure is low**.

**Benefit:**  
- Stabilizes risk estimates while preserving **signal** in high-exposure cells.

---

## Station-Level Risk (Map)

We can aggregate risk over time to get a **station-level** risk measure:

\[
R^{\text{EB}}_s = \frac{\sum_t H_{s,t}}{\sum_t E_{s,t} + \varepsilon}
\quad \text{then smoothed via EB.}
\]

- Stations are then binned into **risk tiers** (e.g., low / medium / high / very high).
- We visualize them on an interactive **folium map**.

<iframe data-src="presentation/assets/html/station_risk_tiers.html" width="100%" height="520px"></iframe>

**Interpretation:**
- Darker / higher tiers indicate **higher expected severity per trip**.
- Highlights **spatial clusters** where CitiBike and an insurer might focus:
  - Infrastructure / awareness.
  - Differentiated pricing or warnings.

---

## Within-Day Variation: Station × Time-of-Day Risk

Risk is not only spatially heterogeneous; it also varies within the day.

For each station \( s \) and time-of-day bin \( d \):

\[
R^{\text{EB}}_{s,d}
= \text{EB-smoothed risk per trip for station } s \text{ and TOD } d.
\]

- This yields a **station × time-of-day** risk surface.
- We again group estimates into **tiers** for interpretation.

<iframe data-src="presentation/assets/html/station_risk_tiers_by_tod.html" width="100%" height="520px"></iframe>

**Use cases:**
- Identify stations that are:
  - Safe during the day but **riskier at night**.
  - Consistently higher risk across all time-of-day bins.
- Enables **contextual in-app messaging**, e.g.:
  - “This station has elevated cyclist risk in the late evening.”

---

## Temporal Risk Patterns (System-Wide)

We can also look at **system-wide** risk aggregated over all stations by **time of day** (and weekday):

<img src="presentation/assets/img/system_temporal_risk.png" width="70%">

**Reading the plot:**
- Morning and evening may show **elevated risk**, aligned with commuting peaks.
- Late evening or night can have **high risk per trip** despite **low volume**:
  - Lower visibility, impaired driving, or riskier riding behavior.

---

## Interpretation & Strategic Implications

- **High-risk stations:**
  - Target for **infrastructure improvements**, **local campaigns**, or **highlighting** in the app.
- **High-risk times:**
  - Inform **time-dependent pricing**, **warnings**, or **soft nudges** (“ride with extra caution at this time”).
- **Station × time-of-day** perspective:
  - Enables very granular **risk-aware product design**:
    - Insurance pricing
    - Recommended routes
    - Operational prioritization (e.g. redistributing bikes away from extremely risky areas at specific times).

**For an insurance partner:**
- Provides a transparent, statistically grounded way to:
  - Price risk **per trip**, **per station**, **per time-of-day**.
  - Build **micro-insurance** products that reflect actual exposure.


---

# 4. Net Flow Prediction (Concept)

---

## Why Predict Net Flow?

- Operational challenge: some stations repeatedly:
  - **Run empty** (no bikes available).
  - **Become full** (no docks available).
- This harms:
  - **Customer experience** (churn, frustration).
  - Potential **insurance value** if riders switch to riskier modes (e.g., cars).

**Net flow prediction**:
- Aim: predict whether a station will be:
  - **Under-supplied** or **over-supplied** in the near future.
- Inputs:
  - Historical **arrivals and departures**.
  - **Calendar features**: day of week, month, holidays.
  - **Lagged** demand and net flow.
  - Potentially **lagged weather**.

---

## From Regression to Ternary Classification

- Net flow is a **continuous variable**:
  - Hard to predict exactly due to noise and shocks.
- For operations, we often only need to know **direction and magnitude** of imbalance.

One approach:
- Define a **tolerance band** around zero (e.g., ±5 bikes).
- Construct a **ternary target**:
  - `-1`: significantly more **departures** than arrivals (likely to run empty).
  - `0`: roughly **balanced**.
  - `+1`: significantly more **arrivals** than departures (likely to become full).

Advantages:
- More **robust** than predicting the exact net flow.
- Directly aligned with **rebalancing decisions**.

*(Details and model results to be added in the future section.)*

---

# 5. Conclusion (Draft)

---

## Key Takeaways (So Far)

1. **CitiBike usage**:
   - Strong **seasonality**, clear **weekday/weekend** and **hour-of-day** patterns.
   - Diverse user base (members vs casual, classic vs e-bikes).

2. **Risk analysis**:
   - Built a **station- and time-specific risk measure**:
     - Collisions assigned via **BallTree** within 300m.
     - Severity per CitiBike trip with **EB smoothing**.
   - Identified **temporal patterns** in risk that do not always mirror demand.

3. **Net flow**:
   - Clear imbalances at some stations.
   - Motivates a **ternary prediction approach** for rebalancing.

---

## Opportunities for CitiBike and an Insurance Partner

- **Micro-insurance products**:
  - Per-ride or membership-based coverage.
  - Pricing informed by **station- and time-specific risk**.
- **User experience enhancements**:
  - In-app **safety information**:
    - “Your current trip starts in a higher-risk area at this time.”
  - **Dynamic routing** suggestions to safer corridors (future extension).
- **Operational benefits**:
  - Better **rebalancing** based on net flow predictions.
  - Reduced service failures, potentially **lower incident rates**.

---

## Next Steps

- Finalize:
  - **Net flow prediction models** (daily or hourly resolution).
  - **Station-level risk maps** for visual communication to stakeholders.
- Integrate:
  - **Lagged weather** and **special events** into both demand and risk models.
- Explore:
  - **Scenario analysis**:
    - How does risk change with more e-bikes?
    - Impact of targeted infrastructure improvements or safety campaigns.

---

## Thank You

Questions?

