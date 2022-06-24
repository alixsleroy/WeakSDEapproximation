# WeakSDEapproximation
This repository follows the work done on weak approximation of SDE.

In the notebook "Infinite double well-histograms.ipynb", I look at the simulation of the infinite double well SDE on large time, using different values of the temperature and the time discretisation. 


In the notebook "infinitedoublewell_eta.ipynb", I characterise the function $\hat{\eta}(t|\tau)=E_{\rho_N(.)} \eta =\int_{b}^\infty \rho_N(x) dx= \mathbb{P}_{\rho_N(.)}(X\geqslant b)$. 

In the notebook "infinitedoublewell_p0x10.ipynb ", I characterise the function $
\mathbb{P}_{\rho_{N_{tot}}(.)}(0 \geqslant X\geqslant b)  \approx \frac{\sum \{ X_{N_{tot}}^{(j)} | X_{N_{tot}}^{(j)} >b \text{ or }X_{N_{tot}}^{(j)} <0 \}}{ \sum_j^{n_{\text{sample}}}  X_{N_{tot}}^{(j)} }$.

Noticing weird behaviour for larger time discretisation $\Delta t$ values, I explore the linearised SDE in "Infinite_doublewell_largedt.ipynb" and try to find why I can observe that behaviour. 
 

In the notebook "Double well SDE.ipynb", I look at the double well SDE.
