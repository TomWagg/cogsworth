---
title: 'cogsworth: A Gala of COSMIC proportions combining binary stellar evolution and galactic dynamics'
tags:
  - Python
  - astronomy
  - binary stellar evolution
  - galactic dynamics
authors:
  - name: Tom Wagg
    orcid: 0000-0001-6147-5761
    affiliation: "1, 2"
  - name: Katelyn Breivik
    orcid: 0000-0001-5228-6598
    affiliation: "3"
  - name: Mathieu Renzo
    orcid: 0000-0002-6718-9472
    affiliation: "2"
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "4"
affiliations:
 - name: Department of Astronomy, University of Washington, Seattle, WA, 98195
   index: 1
 - name: Center for Computational Astrophysics, Flatiron Institute, 162 Fifth Ave, New York, NY, 10010, USA
   index: 2
 - name: McWilliams Center for Cosmology and Astrophysics, Department of Physics, Carnegie Mellon University, Pittsburgh, PA 15213, USA
   index: 3
 - name: University of Arizona, Department of Astronomy \& Steward Observatory, 933 N. Cherry Ave., Tucson, AZ 85721, USA
   index: 4
date: 06 Septemeber 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: #10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: #Astrophysical Journal <- The name of the AAS journal.
---

# Summary

We present `cogsworth`, an open-source Python tool for producing self-consistent population synthesis and galactic dynamics simulations. With `cogsworth` one can (1) sample a population of binaries and star formation history, (2) perform rapid (binary) stellar evolution, (3) integrate orbits through the galaxy and (4) inspect the full evolutionary history of each star or compact object, as well as their positions and kinematics. We include the functionality for post-processing hydrodynamical zoom-in simulations as a basis for galactic potentials and star formation histories to better account for initial spatial stellar clustering and more complex potentials. Alternatively, several analytic models are available for both the potential and star formation history. `cogsworth` can transform the intrinsic simulated population to an observed population through the joint application of dust maps, bolometric correction functions and survey selection functions.

# Statement of need

TODO

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Acknowledgements

We gratefully acknowledge many fruitful discussions with Julianne Dalcanton and Eric Bellm that resulted in several helpful suggestions. TW acknowledges valuable conversations with Matt Orr and Chris Hayward regarding the FIRE simulations, and with Alyson Brooks and Akaxia Cruz regarding the ChaNGa simulations. TW thanks the Simons Foundation, Flatiron Institute and Center for Computational Astrophysics for running the pre-doctoral program during which much of this work was completed. The Flatiron Institute is supported by the Simons Foundation. TW and KB acknowledge support from NASA ATP grant 80NSSC24K0768.

# References