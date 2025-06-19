#### Alpha (α) - Randomization Parameter

- Controls the randomness in firefly movement and helps with exploration vs exploitation balance.
- Typical Range: 0.1 to 1.0 (often starts at 1.0 and decays)

Effects:
- High α (0.5-1.0): Increases exploration, helps escape local optima, but may slow convergence
- Low α (0.1-0.3): Promotes exploitation, faster convergence, but higher risk of premature convergence
- Dynamic α: Often implemented with decay (α = α₀ × θᵗ where θ = 0.9-0.99) to start with exploration and gradually shift to exploitation

#### Beta (β) - Attractiveness Parameter

- Determines the initial attractiveness between fireflies before distance decay.
- Typical Range: 0.1 to 1.0 (β₀ = 1.0 is most common)
- Effects:
  - High β (0.5-1.0): Strong attraction between fireflies, promotes information sharing, faster convergence
  - Low β (0.1-0.3): Weaker attraction, slower convergence but better exploration
  - β₀ = 0: Fireflies move independently (essentially random search)

#### Gamma (γ) - Light Absorption Coefficient

- Controls how quickly attractiveness decreases with distance between fireflies.
- Typical Range: 0.01 to 10 (commonly 0.1 to 1.0)

- Effects:
  - High γ (1.0-10): Rapid decrease in attractiveness with distance, promotes local search, multiple sub swarms
  - Low γ (0.01-0.1): Slower decrease, allows for longer-range exploration, more global search
  - γ = 0: No distance effect, fireflies are only attracted to the brightness one