# FloodRiskConfluences
MSc Thesis: 
Flood risk analysis for river confluences. 
Evaluation of the use of long synthetic time series for the Rhine River
by M.d.S. Fonseca Cerda
under the supervision of Dr.ir.F.Diermanse, Dr.ir.H.Winsemius, Dr.ir.O.Morales, Dr.ir.W.Luxemburg

## Meteorological and Hydrological Evaluation
Evaluation of 74 confluences

How do the different catchment characteristics and the meteorological and hydrological model domains influence the correlations between joining rivers?

For the meteorological domain, three alternatives are evaluated, 1) daily and 2) multiday events, and 3) accounting for the precipitation that falls as snow. One alternative for the hydrological domain is evaluated: annual maximum of daily discharges. For each confluence, the annual maximum peaks are identified out of the 50,000 years of meteorological (precipitation) and hydrological (discharge) synthetic data from GRADE (Hegnauer et al., 2014). Subsequently, the sets of extremes are explored in terms of seasonality, identifying the season where most of the peaks occur per confluence. Moreover, the Spearman rank correlation is calculated for the sets of extremes, between the mainstream peaks and the tributary stream peaks for each confluence, which can be used to evaluate the statistical dependencies between neighbouring catchments.

## Hydraulic Evaluation
Evaluation of Confluence 68, located in the outlet of the Upper Rhine region where the Main River joins the Rhine River

What are the hydraulic interactions between joining rivers?

From the hydrological evaluation, we define the boundary conditions for the hydraulic model. First, the month with the fewest extreme peaks is identified and selected as the start of the hydrological year of the block maxima. Subsequently, the minimum and maximum peak magnitudes of both rivers are identified from the data and selected as the boundaries of a 5x5 grid of discharges, so in total 25 possible combinations of Rhine and Main peak discharges can be used to perform the hydraulic simulations. Finally, representative normalized hydrographs are derived to provide the input time series of the model simulations. In each simulation, 1 day time difference between the 2 river peaks is assumed.
Overall, 25 hydraulic simulations of possible discharge combinations are carried out. The results are evaluated in terms of the maximum water depth (hmax), identifying the differences between each simulation and the hydraulic interactions that take place at the evaluated location. Moreover, two responses function are obtained, which are calculated in terms of the flooded area and the flooded volume as a surrogate for socio-economic impacts.

## Design Flood Event
Evaluation of Confluence 68, located in the outlet of the Upper Rhine region where the Main River joins the Rhine River

How should hydrological events be sampled for performing simulations in a flood risk analysis and/or determining design flood events?

The annual maxima peaks are identified out of the 50,000 years of hydrological (discharge) synthetic data from GRADE. Three different ways of selecting the maxima were evaluated (see MSc Thesis report, Section 2.2.5), leading to three extreme sets (Set 1 | MSmax-TSconc, Set 2 | TSmax-MSconc, and Set 3 | Cmax (C=MS+TS)). From the sets, using the response function, the flooded area that can be expected for certain return periods is calculated, which is used as benchmarks.
Subsequently, random samples are selected from the full data set to resemble commonly recorded data (20, 50, and 100) and possible lengths for generating synthetic data (500, 1000, and 5000). The parameters of the copula functions, when using copulas to construct the joint probability of the joining river discharges, are calculated for each set. Then, using Monte Carlo simulations we sample the two river discharges and calculate the flooded area from the response function. After N Monte Carlo iterations, we construct an empirical distribution of the flooded areas to later estimate the flooded area that corresponds to a certain probability of exceedance (or return period).
The process is repeated 100 times for each sample in a bootstrapping experiment to be able to quantify the variations in the results.
