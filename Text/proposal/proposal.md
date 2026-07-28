---
geometry: margin=0.85in
fontsize: 11pt
---

# Are our map uncertainty statements honest? A test under real sampling designs

**Hugo Rodrigues**, University of Florida, Range Cattle Research and Education Center

## The problem

Soil and vegetation maps are increasingly published with prediction intervals, statements of the form "we are 90% confident the true value lies between 40 and 80 Mg per hectare". Those intervals are almost always checked against data collected the same way, and often in the same places, as the data used to fit the model. That is a weak test. Field sampling is expensive, so real sample locations tend to be clustered, not evenly spread. We do not know how much that clustering damages an interval claiming 90% confidence.

## Why it is worth doing

For point accuracy the field already settled on a standard: estimate ME, RMSE and the modelling efficiency coefficient from an independent probability sample, using design-based estimators (Brus et al., 2011; Wadoux et al., 2021). For uncertainty there is no such standard. Coverage is reported as if it were a property of the model alone. We want to show it belongs to the model and the design together, and that the difference matters. Agencies now attach uncertainty to reported soil carbon stocks, so an interval that is quietly too narrow stops being a statistical detail.

## What we plan to do

1. **Define coverage as a design-based quantity.** Write the probability that an interval contains the truth as a population mean, then estimate it with the Horvitz-Thompson estimator from an independent probability sample.
2. **Build the intervals with CV+** (Barber et al., 2021), which carries a finite-sample guarantee and needs no separate calibration set when the sample is small.
3. **Vary only the sampling design.** Run the same model under simple random, systematic and clustered sampling, so any change in coverage is attributable to the design alone.
4. **Report interval quality beside point accuracy,** using coverage probability and mean interval width alongside ME, RMSE and MEC.
5. **Secondary question.** The variogram is normally fitted by hand, its family, range, binning and anisotropy fixed by the analyst beforehand. We plan to let cross-validation choose them instead, and report honestly whether it helps, since it may not.

## Data, tools and outcome

We start from a published benchmark dataset, so results are comparable with existing work, then apply the same procedure to a rangeland soil property of local interest. Computation runs in *GeoVersa*, an R package we already maintain, keeping the study reproducible from one script. Estimated effort is six months. The output is a short methods paper making one point, that a coverage number without its design is incomplete, plus open code others can reuse. Target journal: *Applied Computing and Geosciences* or *Geoderma*.
