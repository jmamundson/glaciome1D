**glaciome1D** is a quasi-one-dimensional continuum model for ice melange flow. In many respects it is similar to one-dimensional models of ice streams and ice shelves, except that it uses the nonlocal granular fluidity rheology of Henann and Kamrin (2013). The model was created with the intention of developing coupled glacier-ocean-melange models. This is reflected in the modeling framework, which mimics that used for ice streams and ice shelves (Schoof, 2007).

Using the model involves creating an instance of the glaciome class, which contains information on the glacier velocity, viscosity (granular fluidity), and geometry as well as model parameters and various external forcings. The glaciome class includes several basic and easy to use functions, such as: self.diagnostic(), self.prognostic(), self.steadystate(), self.save().



References:
Henann, D. L. and Kamrin, K.: A predictive, size-dependent continuum model for dense granular flows, Proc. Nat. Acad. Sci., 110, 6730–6735, https://doi.org/10.1073/pnas.1219153110, 2013.

Schoof, C.: Ice sheet grounding line dynamics: Steady states, stability, and hysteresis, J. Geophys. Res., 112, F03S28, https://doi.org/10.1029/2006JF000664, 2007.
