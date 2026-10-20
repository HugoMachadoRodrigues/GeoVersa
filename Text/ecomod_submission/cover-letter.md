# Cover letter

<!--
Two slots to fill before sending:
  [EDITOR]  the handling editor's name, if the submission system gives one.
            "Dear Editors" is correct if it does not.
  [DATE]    date of submission.
Everything else is drawn from the manuscript and should not need editing.
-->

[DATE]

[EDITOR]
Editor, *Ecological Modelling*

Dear [EDITOR],

We submit for your consideration the manuscript **"Design-based assessment of
prediction-interval coverage in spatial mapping"**, by Hugo Rodrigues and Golmar
Golmohammadi, as a research article.

Wadoux, Heuvelink, de Bruin and Brus argued in this journal in 2021 that map
accuracy must be estimated from a probability sample by design-based inference,
and that spatial cross-validation has no theoretical standing for that purpose.
Their argument concerns the point map. It leaves open the other half of a map's
quality statement, the prediction interval attached to each value, and that is
where this paper begins.

Our first and principal claim is that the same apparatus applies without any
new theory. Interval coverage is a parameter of the finite population of exactly
the same kind as the mean squared error, so the Horvitz-Thompson argument that
makes accuracy design-unbiased carries over unchanged, and for coverage it is
exactly rather than approximately unbiased. The derivation takes three lines.

The payoff is what it makes visible. On the Amazonian above-ground biomass
benchmark of Wadoux et al. (2021), cross-fitted conformal intervals are mildly
conservative at 0.907 under representative sampling and cover only 0.861 under
the clustered design of that study, a failure no check resting on
exchangeability can detect, because exchangeability is what clustering destroys.
Two results then follow that we did not anticipate. Enlarging the calibration
sample does not repair the shortfall: holding the primary units fixed and
quadrupling the points inside them lowers coverage further, from 0.883 to 0.856,
while the map improves. And recording each validation point's distance to the
nearest calibration point shows why. The realised error grows with that distance
and the interval width does not, and designs whose overall coverage differs by
0.07 cover almost identically at matched distance. A spatial-block calibration
holds 0.93 to 0.95 throughout, at the cost of width.

We think the work suits *Ecological Modelling* because it extends an argument
this journal published and addresses the same readership: the paper is written
as spatial predictive mapping rather than digital soil mapping specifically,
since the derivation applies to any map of a spatial variable built from a point
sample.

We have tried to be disciplined about what the evidence supports. The paper
states that the conformal guarantee is nowhere violated in our experiments and
that what clustering consumes is the empirical margin a practitioner relies on;
that the spatial-block repair restored empirical coverage in this experiment and
carries no theorem; and that selecting the variogram's parameters, as opposed to
representing spatial structure through a variogram at all, buys no detectable
accuracy, which an ablation settles against us. The Limitations section names
the single surface, the remote-sensing benchmark, and the analyses that remain
at one sample size.

The manuscript is original, is not under consideration elsewhere, and has been
approved by both authors. The research received no external funding and we
declare no competing interests. Reference data are the public materials of
Wadoux et al. (2021), obtained from the authors' repository and not
redistributed with our work. The analysis code is openly available and archived
on Zenodo (https://doi.org/10.5281/zenodo.23487542). In accordance with the
journal's policy, the manuscript carries a declaration of the generative AI
tools used in its preparation.

We would be glad to answer any questions, and we thank you for your
consideration.

Yours sincerely,

**Hugo Rodrigues**
University of Florida, Range Cattle Research and Education Center
3401 Experiment Station, Ona, FL 33865, USA
rodrigues.machado.hugo@gmail.com
ORCID 0000-0002-8070-8126

on behalf of Golmar Golmohammadi (ORCID 0000-0001-5532-3892)
