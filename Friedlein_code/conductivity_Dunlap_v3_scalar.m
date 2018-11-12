function sig_cond = conductivity_Dunlap_v3_scalar(eta,sig_DOS,E0,EF,T,n,approx)
%This function calculates the  conductivity of a system according to the
%Dunlap extension of the AHL conductivity model. (see ...
%/Desktop/Organic electronic papers/AHL Conductance - Dave Dunlap.pdf)
%
%In v2, I DON'T normalize energy variables to kT.
%
%In v3, you are only allowed scalars for eta, sig_DOS, and E0. No parallel
%for loop is used.
%
%sig_cond: conductivity of the system, normalized to the maximum
%conductivity, gmax/Rc, where gmax is defined in the abovementioned
%document, and Rc is a characteristic length scale for the system. In
%general, sig_cond is a vector defined such that sig_cond(i) is the
%normalized conductivity when the Fermi level is EF(i).
%
%eta: percolation fraction. Percolative transport will occur when a
%fraction eta of all connections have conductance greater than the critical
%conductance. This must be a scalar.
%
%sig_DOS: width of the standard deviation of the Gaussian DOS. Scalar in J
%
%EO: Energy at which the peak of the DOS is located. Scalar in J
%
%EF: Fermi level (chemical potential) of the system. Vector in J.
%
%T: Temperature of the system. Scalar in K
%
%n: number of discretization steps in the numerical integrals.
%
%approx: flag describing whether to use the approximation or the full
%expression. approx is a boolean scalar. if true, the approximation will be
%used. This input is optional. If you leave it blank, the full expression
%will be used.

%Use the default value for approx if it is not input.
if nargin<7
    approx=boolean(0);
end

%Number of values of EF
n_EF=length(EF);

ymax=zeros(n_EF,1);
%Now loop through the values of EF, and find ymax at each value.
for i=1:n_EF
    ymax(i)=solve_eta_ymax_v2(eta,sig_DOS,E0,EF(i),T,n,approx);
end

%Now use equation 13 to find the normalized conductivity.
sig_cond=2*exp(-abs(ymax))./(cosh(ymax)+1);
