---
title: 'cogsworth: A Gala of COSMIC Proportions Combining Binary Stellar Evolution and Galactic Dynamics'
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
    affiliation: "4"
  - name: Adrian M. Price-Whelan
    orcid: 0000-0003-0872-7098
    affiliation: "2"
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
aas-doi: 10.3847/1538-4365/ad8b1f
aas-journal: Astrophysical Journal Supplement Series
---

# Summary

We present `cogsworth`, an open-source Python tool for producing self-consistent population synthesis and galactic dynamics simulations. With `cogsworth` one can (1) sample a population of binaries and star formation history, (2) perform rapid (binary) stellar evolution, (3) integrate orbits through the galaxy and (4) inspect the full evolutionary history of each star or compact object, as well as its positions and kinematics. We include the functionality for post-processing hydrodynamical zoom-in simulations as a basis for galactic potentials and star formation histories to better account for initial spatial stellar clustering and more complex potentials. Alternatively, several analytical models are available for both the potential and star formation history. `cogsworth` can transform the intrinsic simulated population into an observed population through the joint application of dust maps, bolometric correction functions, and survey selection functions. This paper is jointly published in ApJS [@cogsworth_full].

# Statement of need

The majority of stars are born in binaries and multiple star systems (e.g., @Duchene+2013; @Moe+2017; @Offner+2023), a large subset of which will exchange mass at some point in their lives (e.g., @Podsiadlowski+1992; @Sana+2012; @deMink+2014). These massive stars play a critical role in the formation and evolution of galaxies as a result of their feedback (e.g., @Dekel+1986; @Hopkins+2012; @Nomoto+2013; @Somerville+2015; @Naab+2017). However, binary evolution remains uncertain, with many parameters such as common-envelope efficiency, mass transfer efficiency, angular momentum loss due to mass transfer and the mean magnitude of supernova natal kicks unconstrained over several orders of magnitude (e.g., @Janka+2012; @Ivanova+2013; @Katsuda+2018; @Ivanova+2020; @Ropke+2023; @Marchant2023).

Single massive stars are not expected to migrate far from their birth location before reaching core-collapse due to their short lifetimes ($\lesssim50$\,Myr, e.g., @zapartas:17). However, binary stars may disrupt after an initial supernova event, ejecting the secondary star from the system at its orbital velocity (e.g., @Blaauw+1961; @Eldridge+2011; @Renzo+2019). Thus, close massive binaries that disrupt can lead to the displacement of secondary stars significantly farther from star-forming regions. The present-day positions and kinematics of massive stars and binary products are therefore strongly impacted by changes in binary physics that alter the pre-supernova separation. This means that comparing simulations of positions and kinematics of stars and compact objects to observations will enable constraints on binary stellar evolution parameters.

The use of positions and kinematics as tracers of binary evolution has been considered in the past. Recent work has shown the importance of accounting for the galactic potential, which can change the velocity of kicked objects (e.g. @Disberg+2024). It is also important to consider the inclination or timing of a supernova kick relative to the galactic orbit, since, for example, a kick out of the galactic plane at an object's highest galactic vertical position will have a strong effect on its final position. Failing to consider impacts from both a galactic potential and kicks (i.e. velocity impulses) will lead to misleading conclusions regarding the final spatial distributions of the population. Some studies have considered using the Galactic potential at the present-day positions of objects to place a lower limit on the peculiar velocity at birth and constrain supernova kicks [@Repetto+2012; @Repetto+2015; @Repetto+2017; @atri:19], but the accuracy of this method is debated [@Mandel+2016]. Other works have considered the impact of the Galactic potential for individual special cases, rather than at a population level. For example, @Evans+2020 considered the orbits of hyper-runaway candidates evolving through the Milky Way potential, while @Neuhauser+2020 developed software for tracing the motion of stars to investigate the recent nearby supernovae that ejected $\zeta$ Ophiuchi. @Andrews+2022 considered galactic orbits of synthetic populations to place constraints on black hole natal kicks based on observations of a microlensed black hole.

Additionally, there are several works that consider a full population of objects integrated through a galactic potential. @underworld and @Sweeney+2024 used a combination of `GALAXIA` and `galpy` to predict the spatial distribution of black holes and neutron stars in the Milky Way. Similarly, several works have combined population synthesis with galactic orbit integration (e.g. using `COMPAS` [@COMPAS] and `NIGO` [@NIGO]) to investigate binary neutron stars and pulsars [@Chattopadhyay+2020; @Chattopadhyay+2021; @Gaspari+2024a; @Disberg+2024b; @Song+2024], as well as binary neutron star mergers and short gamma-ray bursts [@Zevin+2020; @Mandhai+2022; @Gaspari+2024b].

There is a clear need for a unified open-source tool that provides the theoretical infrastructure for making predictions for the positions and kinematics of massive stars and compact objects, placing these systems in the context of their host galaxy and its gravitational potential. `cogsworth` fulfills this need, providing a framework for self-consistent population synthesis and galactic dynamics simulations. The code is applicable to a wide range of binary products, both common and rare, from walkaway and runaway stars to X-ray binaries, as well as gravitational-wave and gamma-ray burst progenitors.

# Acknowledgements

We gratefully acknowledge many fruitful discussions with Julianne Dalcanton and Eric Bellm that resulted in several helpful suggestions. TW acknowledges valuable conversations with Matt Orr and Chris Hayward regarding the FIRE simulations, and with Alyson Brooks and Akaxia Cruz regarding the ChaNGa simulations. TW thanks the Simons Foundation, Flatiron Institute and Center for Computational Astrophysics for running the pre-doctoral program during which much of this work was completed. The Flatiron Institute is supported by the Simons Foundation. TW and KB acknowledge support from NASA ATP grant 80NSSC24K0768.

# References