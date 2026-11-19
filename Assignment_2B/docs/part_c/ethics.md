# Ethical concerns (for Part C)

## TBRGS design decisions
- **Coordinate correction:** wrong site positions can send users to the wrong junction. We
  corrected a known shift, but residual errors remain. Who is accountable for location accuracy?
- **Straight-line distance:** underestimates travel time, so users may arrive later than promised.
- **Imputing faulty/missing data:** filling the ~2% gaps and 10 missing days makes predictions look more certain than they are.
- **Training data coverage:** only 40 Boroondara sites, and no data for 2007-2013. The model is not valid outside this area/period.
- **Routing everyone to the fastest route:** can push traffic onto residential streets, and shift congestion and noise to the people living there.
- **Flat 30 s intersection delay:** ignores signal and time-of-day differences, and can bias the results systematically.

## TBRGS+ (mobile app)
- **Location tracking:** continuous location data is sensitive. Consider minimisation, on-device processing, retention limits and explicit consent.
- **User profiles/preferences:** risk of profiling, and of using data for purposes the user didn't agree to.
- **Crowd-sourced feeds:** users' data would shape routes for others; there is a risk of manipulation or fake congestion.
- **Fairness:** better sensor coverage in some areas means better guidance there, which can disadvantage others.
- **Transparency:** users should know a route is a prediction with an error margin, not a guarantee.

## To expand
- Alternatives for each decision and their effects on users and the owner.
- Sources (APA, 5+ with 2+ peer reviewed).
